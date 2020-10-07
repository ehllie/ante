use crate::cache::{ ModuleCache, DefinitionInfoId };
use crate::error::location::{ Location, Locatable };
use crate::parser::ast::{ self, Ast };
use crate::types::pattern::Constructor::*;
use crate::types::{ Type, TypeInfoBody, TypeInfoId };
use crate::util::join_with;

use std::collections::{ HashMap, HashSet };

/// Compiles the given match_expr to a DecisionTree, doing
/// completeness and redundancy checking in the process.
pub fn compile<'c>(match_expr: &ast::Match<'c>, cache: &mut ModuleCache<'c>) -> Option<DecisionTree> {
    let mut matrix = PatternMatrix::from_ast(match_expr, cache, match_expr.location);
    let result = matrix.compile(cache, match_expr.location);

    if result.context.reachable_branches.len() != match_expr.branches.len() {
        for (i, (pattern, _branch)) in match_expr.branches.iter().enumerate() {
            if !result.context.reachable_branches.contains(&i) {
                warning!(pattern.locate(), "Unreachable pattern");
            }
        }
    }

    if result.context.missed_case_count != 0 {
        result.issue_inexhaustive_errors(cache, match_expr.location);
        None
    } else {
        Some(result.tree)
    }
}

#[derive(Debug, Clone, Eq, Hash)]
pub enum VariantTag {
    True,
    False,
    Unit,
    Tuple,
    UserDefined(DefinitionInfoId),

    /// This tag signals pattern matching should give up completeness checking
    /// for this constructor. Integers and floats are most notably translated to
    /// Fail rather than attempting to approximate the types' full ranges.
    Literal(ast::LiteralKind),
}

impl PartialEq for VariantTag {
    fn eq(&self, other: &VariantTag) -> bool {
        use VariantTag::*;
        match (self, other) {
            (UserDefined(a), UserDefined(b)) => a == b,
            (True, True) => true,
            (False, False) => true,
            (Unit, Unit) => true,
            (Tuple, Tuple) => true,
            (Literal(a), Literal(b)) => a == b,
            _ => false,
        }
    }
}

#[derive(Clone)]
enum Constructor {
    /// Any variable pattern, e.g. `a` `b` or `_`
    MatchAll(DefinitionInfoId),

    /// Any constructor followed by a field list.
    /// e.g. `Some 2` -> Variant(1, `Some`, [2])
    ///      `(1, "two")` -> Variant(1, `,`, [1, "two"])
    Variant(VariantTag, PatternStack),
}

impl Constructor {
    fn is_wildcard(&self) -> bool {
        match self {
            MatchAll(_) => true,
            _ => false,
        }
    }

    fn matches(&self, candidate: &VariantTag) -> bool {
        match self {
            MatchAll(_) => true,
            Variant(tag, _) => tag == candidate,
        }
    }

    fn repeat_matchall<'c>(len: usize, fields: &Vec<Vec<DefinitionInfoId>>,
        cache: &mut ModuleCache<'c>, location: Location<'c>) -> Vec<(Constructor, DefinitionInfoId)>
    {
        assert_eq!(fields.len(), len);

        (0..len).map(|i| {
            // Get the nth existing DefinitionInfoId from fields, or if it
            // doesn't already exist, generate a fresh variable.
            let id = fields[i].get(0).copied().unwrap_or_else(||
                cache.push_definition(".repeat_matchall", location)
            );
            (MatchAll(id), id)
        }).collect()
    }

    /// Set's the constructor's id to the first id in new_ids if new_ids is non-empty.
    /// If new_ids is empty, push the constructor's current id instead.
    fn set_id(pair: &mut (Constructor, DefinitionInfoId), new_ids: &mut Vec<DefinitionInfoId>) {
        match new_ids.get_mut(0) {
            Some(new_id) => {
                if let MatchAll(ref mut id) = pair.0 {
                    *id = *new_id;
                }
                pair.1 = *new_id;
            },
            None => new_ids.push(pair.1),
        }
    }

    /// Takes n fields from the contained Variant's pattern stack, using the existing
    /// DefinitionInfoIds from field_ids if possible for each field.
    /// If self is a MatchAll instead, this will generate n MatchAll patterns, again
    /// using the DefinitionInfoIds from field_ids if possible.
    fn take_n_fields<'c>(self, n: usize, field_ids: &mut Vec<Vec<DefinitionInfoId>>,
        cache: &mut ModuleCache<'c>, location: Location<'c>) -> Vec<(Constructor, DefinitionInfoId)>
    {
        match self {
            MatchAll(_) => Constructor::repeat_matchall(n, field_ids, cache, location),
            Variant(_, mut fields) => {
                assert_eq!(fields.0.len(), n);
                assert_eq!(field_ids.len(), n);

                for (field, ids) in fields.0.iter_mut().zip(field_ids.iter_mut()) {
                    Constructor::set_id(field, ids);
                }

                fields.0
            }
        }
    }
}

#[derive(Clone)]
struct PatternStack(Vec<(Constructor, DefinitionInfoId)>);

impl IntoIterator for PatternStack {
    type Item = (Constructor, DefinitionInfoId);
    type IntoIter = std::iter::Rev<std::vec::IntoIter<Self::Item>>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter().rev()
    }
}

impl PatternStack {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn from_ast<'c>(ast: &Ast<'c>, cache: &mut ModuleCache<'c>, location: Location<'c>) -> PatternStack {
        match ast {
            Ast::Variable(variable) => {
                use ast::VariableKind::TypeConstructor;
                let constructor_and_id = match variable.kind {
                    TypeConstructor(_) => {
                        let tag = VariantTag::UserDefined(variable.definition.unwrap());
                        let fields = PatternStack(vec![]);
                        let variable = cache.push_definition(".from_ast.TypeConstructor", location);
                        (Variant(tag, fields), variable)
                    }
                    _ => {
                        let variable = variable.definition.unwrap();
                        (MatchAll(variable), variable)
                    },
                };
                PatternStack(vec![constructor_and_id])
            },
            Ast::Literal(literal) => {
                let fields = PatternStack(vec![]);

                // Only attempt to match bools and unit values. The ranges of all other
                // literal types are too large.
                let tag = match literal.kind {
                    ast::LiteralKind::Bool(b) => if b { VariantTag::True } else { VariantTag::False },
                    ast::LiteralKind::Unit => VariantTag::Unit,
                    _ => VariantTag::Literal(literal.kind.clone()),
                };

                let variable = cache.push_definition(".from_ast.Literal", location);
                PatternStack(vec![(Variant(tag, fields), variable)])
            },
            Ast::Tuple(tuple) => {
                let fields = tuple.elements.iter().rev()
                    .flat_map(|element| PatternStack::from_ast(element, cache, location))
                    .collect();

                let pattern = Variant(VariantTag::Tuple, PatternStack(fields));
                let variable = cache.push_definition(".from_ast.Tuple", location);
                PatternStack(vec![(pattern, variable)])
            },
            Ast::FunctionCall(call) => {
                match call.function.as_ref() {
                    Ast::Variable(variable) => {
                        let tag = VariantTag::UserDefined(variable.definition.unwrap());
                        let fields = call.args.iter().rev()
                            .flat_map(|arg| PatternStack::from_ast(arg, cache, location))
                            .collect();

                        let fields = PatternStack(fields);
                        let variable = cache.push_definition(".from_ast.FunctionCall", location);
                        PatternStack(vec![(Variant(tag, fields), variable)])
                    },
                    _ => {
                        error!(ast.locate(), "Invalid syntax used in pattern");
                        PatternStack(vec![])
                    }
                }
            },
            _ => {
                error!(ast.locate(), "Invalid syntax used in pattern");
                PatternStack(vec![])
            }
        }
    }

    fn head(&self) -> Option<&(Constructor, DefinitionInfoId)> {
        self.0.last()
    }

    fn specialize_row<'c>(&self, tag: &VariantTag, arity: usize, fields: &mut Vec<Vec<DefinitionInfoId>>,
                          cache: &mut ModuleCache<'c>, location: Location<'c>) -> Option<Self> {
        match self.head() {
            Some((head, _)) if head.matches(tag) => {
                let mut new_stack = self.0.clone();

                let (head, _) = new_stack.pop().unwrap();
                new_stack.append(&mut head.take_n_fields(arity, fields, cache, location));
                
                Some(PatternStack(new_stack))
            }
            _ => None,
        }
    }

    /// Given self = [patternN, ..., pattern2, head]
    ///  Return Some [patternN, ..., pattern2]   if head == MatchAll
    ///         None                             otherwise
    fn default_specialize_row(&self) -> Option<(Self, DefinitionInfoId)> {
        self.head().filter(|(constructor, _)| constructor.is_wildcard()).map(|constructor| {
            // Since is_wildcard is true, this if let should always pass
            if let MatchAll(id) = constructor.0 {
                assert_eq!(id, constructor.1);
            }

            let stack = self.0.iter().take(self.0.len() - 1).cloned().collect();
            (PatternStack(stack), constructor.1)
        })
    }
}

fn get_type_info_id(typ: &Type) -> TypeInfoId {
    match typ {
        Type::UserDefinedType(id) => *id,
        Type::TypeApplication(typ, _) => get_type_info_id(typ.as_ref()),
        _ => unreachable!("get_type_info_id called on non-sum-type: {:?}", typ),
    }
}

/// Returns the type that a constructor constructs.
/// Used as a helper function when checking exhaustiveness.
fn get_variant_type_from_constructor<'c>(constructor_id: DefinitionInfoId, cache: &ModuleCache<'c>) -> TypeInfoId {
    let constructor_type = &cache.definition_infos[constructor_id.0].typ;
    match constructor_type {
        Some(Type::ForAll(_, typ)) => {
            match typ.as_ref() {
                Type::Function(_, return_type) => get_type_info_id(return_type.as_ref()),
                typ => get_type_info_id(typ),
            }
        },
        Some(Type::Function(_, return_type)) => get_type_info_id(return_type.as_ref()),
        Some(Type::UserDefinedType(id)) => *id,
        _ => unreachable!("get_variant_type_from_constructor called on invalid constructor of type: {:?}", constructor_type),
    }
}

fn insert_if<T: Eq + std::hash::Hash>(mut set: HashSet<T>, element: T, condition: bool) -> Option<HashSet<T>> {
    if condition {
        set.insert(element);
    }
    Some(set)
}

/// The builtin constructors true, false, and unit don't have DefinitionInfoIds
/// so they must be manually handled here.
fn get_missing_builtin_cases<T>(variants: &HashMap<&VariantTag, T>) -> Option<HashSet<VariantTag>> {
    let mut variants_iter = variants.iter().map(|(tag, _)| tag.clone());
    let (first, second) = (variants_iter.next(), variants_iter.next());
    let missing_cases = HashSet::new();

    use VariantTag::*;
    match (first, second) {
        (Some(True), second)  => insert_if(missing_cases, False, second != Some(&False)),
        (Some(False), second) => insert_if(missing_cases, True,  second != Some(&True)),
        (Some(Unit), _) => Some(missing_cases),
        (Some(Tuple), _) => Some(missing_cases),
        // Literals always require a match-all, so a missing case is always inserted here.
        (Some(Literal(literal)), _) => insert_if(missing_cases, Literal(literal.clone()), true),
        _ => None,
    }
}

fn get_covered_constructors<T>(variants: &HashMap<&VariantTag, T>) -> HashSet<VariantTag> {
    variants.iter().map(|(tag, _)| (*tag).clone()).collect()
}

/// Given a hashmap from variant tag -> arity,
/// return true if the hashmap covers all constructors for its type.
fn get_missing_cases<'c, T>(variants: &HashMap<&VariantTag, T>, cache: &ModuleCache<'c>) -> HashSet<VariantTag> {
    use VariantTag::*;

    if let Some(result) = get_missing_builtin_cases(variants) {
        return result;
    }

    match variants.iter().nth(0).map(|(tag, _)| tag.clone()).unwrap() {
        True | False | Unit | Tuple | Literal(_) =>
            unreachable!("Found builtin constructor not covered by builtin_is_exhastive"),

        UserDefined(id) => {
            let type_id = get_variant_type_from_constructor(*id, cache);
            match &cache.type_infos[type_id.0].body {
                TypeInfoBody::Union(constructors) => {
                    let all_constructors: HashSet<_> = constructors.iter().map(|constructor| VariantTag::UserDefined(constructor.id)).collect();
                    let covered_constructors = get_covered_constructors(variants);
                    all_constructors.difference(&covered_constructors).cloned().collect()
                },

                // Structs only have one constructor anyway, so if
                // we have a constructor its always exhaustive.
                TypeInfoBody::Struct(_) => HashSet::new(),
                TypeInfoBody::Alias(_) => unimplemented!("Pattern matching on aliased types is unimplemented"),
                TypeInfoBody::Unknown => unreachable!("Cannot pattern match on unknown type constructor"),
            }
        }
    }
}

/// Given a set of matching constructors:
///   C a b c
///   C d (C2 e) f
///   C () g 5
/// collect the variables bound to each field:
///   [ [a, d], [b, g], [c, f] ]
fn collect_fields(rows: Vec<&PatternStack>) -> Vec<Vec<DefinitionInfoId>> {
    let mut variables = vec![];

    for col in 0 .. rows[0].len() {
        variables.push(vec![]);

        for row in rows.iter().copied() {
            if let Some((MatchAll(id), _)) = row.0.get(col) {
                variables[col].push(*id);
            }
        }
    }

    variables
}

#[derive(Default)]
struct PatternMatrix {
    /// Each row holds the pattern stack of the pattern for a particular branch as
    /// well as the index of the branch that the pattern leads to in the source if matched.
    rows: Vec<(PatternStack, usize)>,
}

impl PatternMatrix {
    fn from_ast<'c>(match_expr: &ast::Match<'c>, cache: &mut ModuleCache<'c>, location: Location<'c>) -> PatternMatrix {
        let rows = match_expr.branches.iter().enumerate()
            .map(|(branch_index, (pattern, _))|
                 (PatternStack::from_ast(pattern, cache, location), branch_index))
            .collect();

        PatternMatrix { rows }
    }

    /// This function corresponds to S(c, P -> A) in "Compiling
    /// Pattern Matching to Good Decision Trees"
    ///
    /// Remove all rows of the matrix whose first pattern is not the
    /// given construct or a catch-all pattern. For the rows that are
    /// retained, pop the first pattern off the stack for that row. E.g.
    ///
    /// Matrix =
    ///     []      _      -> 1
    ///     _       []     -> 2
    ///     (_::_)  (_::_) -> 3
    ///
    /// specialize(Matrix, (::)) =
    ///     _ _ []      -> 2
    ///     _ _ (_::_)  -> 3
    ///
    fn specialize<'c>(&self, tag: &VariantTag, arity: usize, fields: &mut Vec<Vec<DefinitionInfoId>>,
        cache: &mut ModuleCache<'c>, location: Location<'c>) -> Self
    {
        let mut matrix = PatternMatrix::default();

        for (row, branch) in self.rows.iter() {
            match row.specialize_row(tag, arity, fields, cache, location) {
                Some(row) => matrix.rows.push((row, *branch)),
                None => (),
            }
        }

        matrix
    }

    /// This function corresponds to D(P -> A) in "Compiling
    /// Pattern Matching to Good Decision Trees"
    ///
    /// Remove all rows of the matrix whose first pattern is
    /// not a catch-all. For the rows that are retained, pop
    /// the catch-all off the stack for that row. E.g.
    ///
    /// Matrix =
    ///     [] _  -> 1
    ///     _  [] -> 2
    ///     _  _  -> 3
    ///
    /// default_specialize(Matrix) =
    ///     [] -> 2
    ///     _  -> 3
    ///
    /// This function is changed slightly to also collect the
    /// variables used, then compile the resulting matrix into
    /// a decision tree.
    fn default_specialize<'c>(&self, cache: &mut ModuleCache<'c>, location: Location<'c>) -> (DecisionTreeResult, Vec<DefinitionInfoId>) {
        let mut matrix = PatternMatrix::default();
        let mut variables_to_bind = vec![];

        for (row, branch) in self.rows.iter() {
            match row.default_specialize_row() {
                Some((row, variable_id)) => {
                    matrix.rows.push((row, *branch));
                    variables_to_bind.push(variable_id);
                },
                None => (),
            }
        }

        (matrix.compile(cache, location), variables_to_bind)
    }

    /// Generate a Switch branch covering each case of the top pattern on the stack.
    /// Handles exhaustiveness checking for the union internally.
    fn switch_on_pattern<'c>(&mut self, cache: &mut ModuleCache<'c>, location: Location<'c>) -> DecisionTreeResult {
        // Generate the set of constructors appearing in the column
        let mut matched_variants = HashMap::new();
        let mut switching_on = None;

        for (row, _) in self.rows.iter() {
            if let Some((Variant(tag, fields), var)) = row.head() {
                switching_on = Some(*var);

                matched_variants.entry(tag)
                    .or_insert(vec![])
                    .push(fields);
            }
        }

        let missed_cases = get_missing_cases(&matched_variants, cache);
        let mut context = DecisionTreeContext::default();

        let mut cases: Vec<_> = matched_variants.into_iter().map(|(tag, fields)| {
            let arity = fields[0].len();
            let mut fields = collect_fields(fields);

            let branch = self.specialize(tag, arity, &mut fields, cache, location)
                .compile(cache, location);

            // PatternStacks store patterns in reverse order for faster prepending.
            // Reversing fields here undoes this so that only the natural order is
            // stored in the DecisionTree.
            fields.reverse();
            Case {
                tag: Some(tag.clone()),
                fields,
                branch: context.merge(branch),
            }
        }).collect();

        // If we don't have an exhaustive match, generate a default matrix
        if !missed_cases.is_empty() {
            let (branch, fields) = self.default_specialize(cache, location);
            switching_on = fields.get(0).copied().or(switching_on);
            cases.push(Case {
                tag: None,
                fields: vec![fields],
                branch: context.merge(branch),
            });
        }

        let tree = DecisionTree::Switch(switching_on.unwrap(), cases);
        DecisionTreeResult::new(tree, context)
    }

    fn find_first_non_default_column(&self) -> Option<usize> {
        let len = self.rows[0].0.len();

        for col in (1 .. len).rev() {
            for (row, _) in self.rows.iter() {
                match row.0.get(col) {
                    Some((MatchAll(_), _)) => continue,
                    _ => return Some(col),
                }
            }
        }
        None
    }

    fn swap_column<'c>(&mut self, column: usize, cache: &mut ModuleCache<'c>, location: Location<'c>) -> DecisionTreeResult {
        for (row, _) in self.rows.iter_mut() {
            row.0.swap(0, column);
        }

        self.compile(cache, location)
    }

    fn first_row_is_all_wildcards(&mut self) -> bool {
        (self.rows[0].0).0.iter()
            .all(|(constructor, _)| constructor.is_wildcard())
    }

    fn compile<'c>(&mut self, cache: &mut ModuleCache<'c>, location: Location<'c>) -> DecisionTreeResult {
        if self.rows.is_empty() {
            // We have an in-exhaustive case expression
            DecisionTreeResult::fail()
        } else if self.first_row_is_all_wildcards() {
            // If every pattern in the first row is a wildcard it must match.
            DecisionTreeResult::leaf(self.rows[0].1)
        } else {
            // There's at least one non-wild pattern in the matrix somewhere
            for (row, _) in self.rows.iter() {
                match row.head() {
                    Some((Variant(..), _)) => return self.switch_on_pattern(cache, location),
                    Some((MatchAll(_), _)) => continue,
                    None => unreachable!("PatternMatrix rows cannot be empty"),
                }
            }

            // The first column contains only wildcard patterns. Search the
            // matrix until we find a column that has a non-wildcard pattern,
            // and swap columns with column 0
            match self.find_first_non_default_column() {
                Some(column) => self.swap_column(column, cache, location),
                None => self.default_specialize(cache, location).0,
            }
        }
    }
}

/// DecisionTreeResult augments a DecisionTree with information about which cases
/// are redundant (can never be matched) and if the match as a whole is exhaustive or not.
/// Since these extra properties are only used for error-reporting, they may safely be discarded
/// for access to the tree if desired.
struct DecisionTreeResult {
    tree: DecisionTree,
    context: DecisionTreeContext
}

#[derive(Default)]
struct DecisionTreeContext {
    reachable_branches: HashSet<usize>,
    missed_case_count: usize,
}

impl DecisionTreeContext {
    fn merge(&mut self, result: DecisionTreeResult) -> DecisionTree {
        self.missed_case_count += result.context.missed_case_count;
        self.reachable_branches =
            self.reachable_branches.union(&result.context.reachable_branches)
            .copied().collect();

        result.tree
    }
}

impl DecisionTreeResult {
    fn new(tree: DecisionTree, context: DecisionTreeContext) -> DecisionTreeResult {
        DecisionTreeResult { tree, context }
    }

    fn fail() -> DecisionTreeResult {
        let mut context = DecisionTreeContext::default();
        context.missed_case_count = 1;
        DecisionTreeResult::new(DecisionTree::Fail, context)
    }

    fn leaf(branch: usize) -> DecisionTreeResult {
        let mut context = DecisionTreeContext::default();
        context.reachable_branches.insert(branch);
        DecisionTreeResult::new(DecisionTree::Leaf(branch), context)
    }

    fn issue_inexhaustive_errors<'c>(&self, cache: &ModuleCache<'c>, location: Location<'c>){
        let mut bindings = HashMap::new();
        DecisionTreeResult::issue_inexhaustive_errors_helper(&self.tree, None, &mut bindings, cache, location);
    }

    fn issue_inexhaustive_errors_helper<'c>(tree: &DecisionTree, starting_id: Option<DefinitionInfoId>,
        bindings: &mut DebugMatchBindings, cache: &ModuleCache<'c>, location: Location<'c>)
    {
        use DecisionTree::*;
        match tree {
            Leaf(_) => (),
            Fail => unreachable!("DecisionTree::Fail case should be matched on within DecisionTree::Switch"),
            Switch(id, cases) => {
                for case in cases.iter() {
                    match &case.branch {
                        Fail => {
                            let covered_cases = cases.iter()
                                .filter_map(|case| case.tag.as_ref())
                                .map(|tag| (tag, ()))
                                .collect();

                            for tag in get_missing_cases(&covered_cases, cache) {
                                bindings.insert(*id, DebugConstructor::new(&Some(tag), cache));
                                DecisionTreeResult::issue_inexhaustive_error(starting_id, bindings, location);
                            }
                        },
                        _ => {
                            bindings.insert(*id, DebugConstructor::from_case(case, cache));
                            let starting_id = starting_id.or(Some(*id));
                            DecisionTreeResult::issue_inexhaustive_errors_helper(
                                &case.branch, starting_id, bindings, cache, location);
                        }
                    }
                }
                bindings.remove(id);
            },
        }
    }

    fn issue_inexhaustive_error<'c>(starting_id: Option<DefinitionInfoId>,
        bindings: &DebugMatchBindings, location: Location<'c>)
    {
        let case = starting_id.map_or("_".to_string(), |id|
            DecisionTreeResult::construct_missing_case_string(id, bindings));

        error!(location, "Missing case {}", case);
    }

    fn construct_missing_case_string(id: DefinitionInfoId, bindings: &DebugMatchBindings) -> String {
        match bindings.get(&id) {
            None => "_".to_string(),
            Some(case) => {
                let mut case_string = case.tag.clone();
                let case_is_tuple = case.tag == "(";

                // Parenthesizes an argument string if it contains spaces and it's not a tuple field
                let parenthesize = |field_string: String| {
                    if field_string.contains(" ") && !case_is_tuple {
                        format!("({})", field_string)
                    } else {
                        field_string
                    }
                };

                // MatchAlls have fields referencing themselves, skip iterating on
                // their fields to avoid infinite recursion
                if case.tag != "_" {
                    let fields: Vec<String> = case.fields.iter().map(|field| {
                        field.iter()
                            .map(|id| DecisionTreeResult::construct_missing_case_string(*id, bindings))
                            .filter(|field_string| field_string != "_")
                            .nth(0)
                            .map(parenthesize)
                            .unwrap_or("_".to_string())
                    }).collect();

                    if !case_is_tuple {
                        if !fields.is_empty() {
                            case_string = format!("{} {}", case_string, join_with(&fields, " "));
                        } else {
                            case_string = format!("{}", case_string);
                        }
                    } else if fields.len() == 1 {
                        // single-arg tuple. Add trailing ,
                        case_string = format!("({},)", fields[0]);
                    } else {
                        case_string = format!("({})", join_with(&fields, ", "));
                    }
                }

                case_string
            },
        }
    }
}

pub enum DecisionTree {
    /// Success! run the code at the given numerical branch
    Leaf(usize),

    /// The pattern failed to match anything,
    /// if this is constructed we issue a compile time error
    /// that the match is non-exhaustive.
    Fail,

    /// Switch on the given pattern for each case of a tagged union or literal
    Switch(DefinitionInfoId, Vec<Case>),
}

pub struct Case {
    /// The constructor's tag to match on. If this is a match-all case, it is None.
    tag: Option<VariantTag>,

    /// Each field is a Vec of variables to bind to since their can potentially be multiple
    /// names for the same field across different source branches while pattern matching.
    fields: Vec<Vec<DefinitionInfoId>>,

    /// The branch to take in this tree if this constructor's tag is matched.
    branch: DecisionTree,
}

/// Used for bindings values to ids when constructing missing cases to use in errors
type DebugMatchBindings = HashMap<DefinitionInfoId, DebugConstructor>;

struct DebugConstructor {
    /// String form of a Case.tag
    tag: String,

    /// Copied directly from Case.fields.
    /// This could be a reference to avoid cloning, but DebugConstructors should only
    /// be constructed in an error case when a match is inexhaustive anyway.
    fields: Vec<Vec<DefinitionInfoId>>,
}

impl DebugConstructor {
    fn new<'c>(tag: &Option<VariantTag>, cache: &ModuleCache<'c>) -> DebugConstructor {
        use VariantTag::*;
        use crate::parser::ast::LiteralKind;
        let tag = match &tag {
            Some(UserDefined(id)) => {
                cache.definition_infos[id.0].name.clone()
            },
            Some(Literal(LiteralKind::Integer(_))) => "_ : int".to_string(),
            Some(Literal(LiteralKind::Float(_))) => "_ : float".to_string(),
            Some(Literal(LiteralKind::String(_))) => "_ : string".to_string(),
            Some(Literal(LiteralKind::Char(_))) => "_ : char".to_string(),
            Some(Literal(LiteralKind::Bool(_))) => unreachable!(),
            Some(Literal(LiteralKind::Unit)) => unreachable!(),
            Some(True) => "true".to_string(),
            Some(False) => "false".to_string(),
            Some(VariantTag::Unit) => "()".to_string(),
            Some(Tuple) => "(".to_string(),
            None => "_".to_string(),
        };

        DebugConstructor { tag, fields: vec![] }
    }

    fn from_case<'c>(case: &Case, cache: &ModuleCache<'c>) -> DebugConstructor {
        let mut constructor = DebugConstructor::new(&case.tag, cache);
        constructor.fields = case.fields.clone();
        constructor
    }
}

// The rest of the file is pretty-printing for debugging pattern trees.
// Without these impls, DecisionTree's and PatternMatrices can be difficult
// to debug since the naive derive(Debug) takes up to much space to be useful.

impl std::fmt::Debug for DecisionTree {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        fmt_tree(self, f, 0)
    }
}

fn fmt_tree(tree: &DecisionTree, f: &mut std::fmt::Formatter, indent_level: usize) -> Result<(), std::fmt::Error> {
    use DecisionTree::*;
    match tree {
        Leaf(branch) => write!(f, "Leaf({})", branch),
        Fail => write!(f, "Fail"),
        Switch(id, cases) => {
            write!(f, "match ${} with", id.0)?;
            let spaces = " ".repeat(indent_level);
            for case in cases.iter() {
                write!(f, "\n{}| ", spaces)?;
                match &case.tag {
                    Some(VariantTag::Literal(literal)) => write!(f, "{:?}", literal)?,
                    Some(tag) => write!(f, "{:?}", tag)?,
                    None => write!(f, "_")?,
                }

                for field_ids in case.fields.iter() {
                    write!(f, " {:?}", field_ids)?;
                }

                write!(f, " => ")?;
                fmt_tree(&case.branch, f, indent_level + 2)?;
            }
            Ok(())
        }
    }
}

impl std::fmt::Debug for Constructor {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        match self {
            MatchAll(id) => write!(f, "${}", id.0),
            Variant(tag, stack) => {
                if !stack.0.is_empty() {
                    write!(f, "(")?;
                }

                match tag {
                    VariantTag::UserDefined(id) => write!(f, "${}", id.0)?,
                    _ => write!(f, "{:?}", tag)?,
                }

                for (constructor, _) in stack.0.iter().rev() {
                    write!(f, " {:?}", constructor)?;
                }

                if !stack.0.is_empty() {
                    write!(f, ")")?;
                }

                Ok(())
            }
        }
    }
}

impl std::fmt::Debug for PatternMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        write!(f, "matrix = [")?;
        for (row, branch) in self.rows.iter() {
            write!(f, "\n| {:?} => {:?}", row, branch)?;
        }
        write!(f, "\n]")
    }
}

impl std::fmt::Debug for PatternStack {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        write!(f, "[")?;
        for (constructor, id) in self.0.iter().rev() {
            write!(f, " ({:?} as ${})", constructor, id.0)?;
        }
        write!(f, " ]")
    }
}
