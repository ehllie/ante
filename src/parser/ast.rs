//! parser/ast.rs - Defines the abstract syntax tree (Ast)
//! used to hold the source program. This syntax tree is
//! produced as a result of parsing and is used in every
//! subsequent pass.
//!
//! Design-wise, instead of producing a new Ast with the
//! results of a given compiler pass (e.g. returning a TypedAst
//! as the result of type inference that is the same as Ast but
//! with an additional Type field for each node) ante instead
//! uses Option fields and mutably fills in this missing values.
//! For example:
//! - Name resolution fills out all these fields for various types:
//!   - For `ast::Variable`s:
//!       `definition: Option<DefinitionInfoId>`,
//!       `impl_scope: Option<ImplScopeId>,
//!       `id: Option<VariableId>`,
//!   - `level: Option<LetBindingLevel>` for
//!       `ast::Definition`s, `ast::TraitDefinition`s, and `ast::Extern`s,
//!   - `info: Option<DefinitionInfoId>` for `ast::Definition`s,
//!   - `type_info: Option<TypeInfoId>` for `ast::TypeDefinition`s,
//!   - `trait_info: Option<TraitInfoId>` for `ast::TraitDefinition`s and `ast::TraitImpl`s
//!   - `module_id: Option<ModuleId>` for `ast::Import`s,
//!
//! - Type inference fills out:
//!   `typ: Option<Type>` for all nodes,
//!   `decision_tree: Option<DecisionTree>` for `ast::Match`s
use crate::cache::{DefinitionInfoId, EffectInfoId, ImplInfoId, ImplScopeId, ModuleId, TraitInfoId, VariableId};
use crate::error::location::{Locatable, Location};
use crate::lexer::token::{FloatKind, IntegerKind, Token};
use crate::types::pattern::DecisionTree;
use crate::types::traits::RequiredTrait;
use crate::types::typechecker::TypeBindings;
use crate::types::{self, LetBindingLevel, TypeInfoId};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::rc::Rc;

#[derive(Clone, Debug, Eq, PartialOrd, Ord)]
pub enum LiteralKind {
    Integer(u64, Option<IntegerKind>),
    Float(u64, Option<FloatKind>),
    String(String),
    Char(char),
    Bool(bool),
    Unit,
}

#[derive(Debug, Clone)]
pub struct Literal {
    pub kind: LiteralKind,
    pub location: Location,
    pub typ: Option<types::Type>,
}

#[derive(Debug, PartialEq, Clone)]
pub enum VariableKind {
    Identifier(String),
    Operator(Token),
    TypeConstructor(String),
}

/// a, b, (+), Some, etc.
#[derive(Debug, Clone)]
pub struct Variable {
    pub kind: VariableKind,
    pub location: Location,

    /// module prefix path
    pub module_prefix: Vec<String>,

    /// A variable's definition is initially undefined.
    /// During name resolution, every definition is filled
    /// out - becoming Some(id)
    pub definition: Option<DefinitionInfoId>,

    /// The module this Variable is contained in. Determines which
    /// impls are visible to it during type inference.
    pub impl_scope: Option<ImplScopeId>,

    /// The mapping used to instantiate the definition type of this
    /// variable into a monotype, if any.
    pub instantiation_mapping: Rc<TypeBindings>,

    /// A unique ID that can be used to identify this variable node
    pub id: Option<VariableId>,

    pub typ: Option<types::Type>,
}

// TODO: Remove. This is only used for experimenting with ante-lsp
// which does not refer to the instantiation_mapping field at all.
unsafe impl Send for Variable {}
unsafe impl Sync for Variable {}

/// Maps DefinitionInfoIds closed over in the environment to their new
/// IDs within the closure which shadow their previous definition.
/// These new IDs may be instantiations of a type that was generalized
/// (but is now bound to a concrete type as a function parameter as the new id),
/// so we need to remember these instatiation bindings as well.
///
/// Needed because closure environment variables are converted to
/// parameters of the function which need separate IDs.
pub type ClosureEnvironment = BTreeMap<
    DefinitionInfoId,
    (
        /*Confusing: This is a variable id for the DefinitionInfoId key, used for trait dispatch.*/
        VariableId,
        DefinitionInfoId,
        Rc<TypeBindings>,
    ),
>;

/// \a b. expr
/// Function definitions are also desugared to a ast::Definition with a ast::Lambda as its body
#[derive(Debug, Clone)]
pub struct Lambda {
    pub args: Vec<Ast>,
    pub body: Box<Ast>,
    pub return_type: Option<Type>,

    pub closure_environment: ClosureEnvironment,

    pub required_traits: Vec<RequiredTrait>,

    pub location: Location,
    pub typ: Option<types::Type>,
}

// TODO: Remove. This is only used for experimenting with ante-lsp
// which does not refer to the instantiation_mapping field at all.
unsafe impl Send for Lambda {}
unsafe impl Sync for Lambda {}

/// foo a b c
#[derive(Debug, Clone)]
pub struct FunctionCall {
    pub function: Box<Ast>,
    pub args: Vec<Ast>,
    pub location: Location,
    pub typ: Option<types::Type>,
}

impl<'a> FunctionCall {
    pub fn is_pair_constructor(&self) -> bool {
        if let Ast::Variable(variable) = self.function.as_ref() {
            variable.kind == VariableKind::Operator(Token::Comma)
        } else {
            false
        }
    }
}

/// foo = 23
/// pattern a b = expr
#[derive(Debug, Clone)]
pub struct Definition {
    pub pattern: Box<Ast>,
    pub expr: Box<Ast>,
    pub mutable: bool,
    pub location: Location,
    pub level: Option<LetBindingLevel>,
    pub info: Option<DefinitionInfoId>,
    pub typ: Option<types::Type>,
}

/// if condition then expression else expression
#[derive(Debug, Clone)]
pub struct If {
    pub condition: Box<Ast>,
    pub then: Box<Ast>,
    pub otherwise: Box<Ast>,
    pub location: Location,
    pub typ: Option<types::Type>,
}

/// match expression
/// | pattern1 -> branch1
/// | pattern2 -> branch2
/// ...
/// | patternN -> branchN
#[derive(Debug, Clone)]
pub struct Match {
    pub expression: Box<Ast>,
    pub branches: Vec<(Ast, Ast)>,

    /// The decision tree is outputted from the completeness checking
    /// step and is used during codegen to efficiently compile each pattern branch.
    pub decision_tree: Option<DecisionTree>,

    pub location: Location,
    pub typ: Option<types::Type>,
}

/// Type nodes in the AST, different from the representation of types during type checking.
/// PointerType and potentially UserDefinedType are actually type constructors
#[derive(Debug, Clone)]
#[allow(clippy::enum_variant_names)]
pub enum Type {
    // Optional IntegerKind, None = polymorphic int
    Integer(Option<IntegerKind>, Location),
    // Optional FloatKind, None = polymorphic float
    Float(Option<FloatKind>, Location),
    Char(Location),
    String(Location),
    Pointer(Location),
    Boolean(Location),
    Unit(Location),
    Reference(Location),
    Function(Vec<Type>, Box<Type>, /*varargs:*/ bool, /*closure*/ bool, Location),
    TypeVariable(String, Location),
    UserDefined(String, Location),
    TypeApplication(Box<Type>, Vec<Type>, Location),
    Pair(Box<Type>, Box<Type>, Location),
}

/// The AST representation of a trait usage.
/// A trait's definition would be a TraitDefinition node.
/// This struct is used in e.g. `given` to list the required traits.
#[derive(Debug, Clone)]
pub struct Trait {
    pub name: String,
    pub args: Vec<Type>,
    pub location: Location,
}

#[derive(Debug, Clone)]
pub enum TypeDefinitionBody {
    Union(Vec<(String, Vec<Type>, Location)>),
    Struct(Vec<(String, Type, Location)>),
    Alias(Type),
}

/// type Name arg1 arg2 ... argN = definition
#[derive(Debug, Clone)]
pub struct TypeDefinition {
    pub name: String,
    pub args: Vec<String>,
    pub definition: TypeDefinitionBody,
    pub location: Location,
    pub type_info: Option<TypeInfoId>,
    pub typ: Option<types::Type>,
}

/// lhs : rhs
#[derive(Debug, Clone)]
pub struct TypeAnnotation {
    pub lhs: Box<Ast>,
    pub rhs: Type,
    pub location: Location,
    pub typ: Option<types::Type>,
}

/// import Path1 . Path2 ... PathN
#[derive(Debug, Clone)]
pub struct Import {
    pub path: Vec<String>,
    pub location: Location,
    pub typ: Option<types::Type>,
    pub module_id: Option<ModuleId>,
    pub symbols: HashSet<String>,
}

/// trait Name arg1 arg2 ... argN -> fundep1 fundep2 ... fundepN with
///     declaration1
///     declaration2
///     ...
///     declarationN
#[derive(Debug, Clone)]
pub struct TraitDefinition {
    pub name: String,
    pub args: Vec<String>,
    pub fundeps: Vec<String>,

    // Storing function declarations as TypeAnnotations here
    // throws away any names given to parameters. In practice
    // this shouldn't matter until refinement types are implemented
    // that can depend upon these names.
    pub declarations: Vec<TypeAnnotation>,
    pub level: Option<LetBindingLevel>,
    pub location: Location,
    pub trait_info: Option<TraitInfoId>,
    pub typ: Option<types::Type>,
}

/// impl TraitName TraitArg1 TraitArg2 ... TraitArgN
///     definition1
///     definition2
///     ...
///     definitionN
#[derive(Debug, Clone)]
pub struct TraitImpl {
    pub trait_name: String,
    pub trait_args: Vec<Type>,
    pub given: Vec<Trait>,

    pub definitions: Vec<Definition>,
    pub location: Location,
    pub trait_info: Option<TraitInfoId>,
    pub impl_id: Option<ImplInfoId>,
    pub typ: Option<types::Type>,
    pub trait_arg_types: Vec<types::Type>, // = fmap(trait_args, convert_type)
}

/// return expression
#[derive(Debug, Clone)]
pub struct Return {
    pub expression: Box<Ast>,
    pub location: Location,
    pub typ: Option<types::Type>,
}

/// statement1
/// statement2
/// ...
/// statementN
#[derive(Debug, Clone)]
pub struct Sequence {
    pub statements: Vec<Ast>,
    pub location: Location,
    pub typ: Option<types::Type>,
}

/// extern declaration
/// // or
/// extern
///     declaration1
///     declaration2
///     ...
///     declarationN
#[derive(Debug, Clone)]
pub struct Extern {
    pub declarations: Vec<TypeAnnotation>,
    pub level: Option<LetBindingLevel>,
    pub location: Location,
    pub typ: Option<types::Type>,
}

/// lhs.field
#[derive(Debug, Clone)]
pub struct MemberAccess {
    pub lhs: Box<Ast>,
    pub field: String,
    pub location: Location,
    /// True if this is an offset .& operation
    pub is_offset: bool,
    pub typ: Option<types::Type>,
}

/// lhs := rhs
#[derive(Debug, Clone)]
pub struct Assignment {
    pub lhs: Box<Ast>,
    pub rhs: Box<Ast>,
    pub location: Location,
    pub typ: Option<types::Type>,
}

/// effect Name arg1 arg2 ... argN with
///     declaration1
///     declaration2
///     ...
///     declarationN
#[derive(Debug, Clone)]
pub struct EffectDefinition {
    pub name: String,
    pub args: Vec<String>,

    pub declarations: Vec<TypeAnnotation>,
    pub level: Option<LetBindingLevel>,
    pub location: Location,
    pub effect_info: Option<EffectInfoId>,
    pub typ: Option<types::Type>,
}

/// handle expression
/// | pattern1 -> branch1
/// | pattern2 -> branch2
/// ...
/// | patternN -> branchN
///
/// Handle expressions desugar to 1 case per
/// effect or `return`, with any nested patterns
/// deferring to match expressions.
#[derive(Debug, Clone)]
pub struct Handle {
    pub expression: Box<Ast>,
    pub branches: Vec<(Ast, Ast)>,

    /// IDs for each 'resume' variable (1 per branch) of this handle expression.
    /// This is filled out during name resolution.
    pub resumes: Vec<DefinitionInfoId>,

    pub location: Location,
    pub typ: Option<types::Type>,
}

/// MyStruct with
///     field1 = expr1
///     field2 = expr2
#[derive(Debug, Clone)]
pub struct NamedConstructor {
    pub constructor: Box<Ast>,
    pub args: Vec<(String, Ast)>,

    pub location: Location,
    pub typ: Option<types::Type>,
}

#[derive(Debug, Clone)]
pub enum Ast {
    Literal(Literal),
    Variable(Variable),
    Lambda(Lambda),
    FunctionCall(FunctionCall),
    Definition(Definition),
    If(If),
    Match(Match),
    TypeDefinition(TypeDefinition),
    TypeAnnotation(TypeAnnotation),
    Import(Import),
    TraitDefinition(TraitDefinition),
    TraitImpl(TraitImpl),
    Return(Return),
    Sequence(Sequence),
    Extern(Extern),
    MemberAccess(MemberAccess),
    Assignment(Assignment),
    EffectDefinition(EffectDefinition),
    Handle(Handle),
    NamedConstructor(NamedConstructor),
}

unsafe impl Send for Ast {}

impl PartialEq for LiteralKind {
    /// Ignoring any type tags, are these literals equal?
    fn eq(&self, other: &Self) -> bool {
        use LiteralKind::*;
        match (self, other) {
            (Integer(x, _), Integer(y, _)) => x == y,
            (Float(x, _), Float(y, _)) => x == y,
            (String(x), String(y)) => x == y,
            (Char(x), Char(y)) => x == y,
            (Bool(x), Bool(y)) => x == y,
            (Unit, Unit) => true,
            _ => false,
        }
    }
}

impl std::hash::Hash for LiteralKind {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
        match self {
            LiteralKind::Integer(x, _) => x.hash(state),
            LiteralKind::Float(x, _) => x.hash(state),
            LiteralKind::String(x) => x.hash(state),
            LiteralKind::Char(x) => x.hash(state),
            LiteralKind::Bool(x) => x.hash(state),
            LiteralKind::Unit => (),
        }
    }
}

/// These are all convenience functions for creating various Ast nodes from the parser
impl Ast {
    pub fn get_operator(self) -> Option<Token> {
        match self {
            Ast::Variable(variable) => match variable.kind {
                VariableKind::Operator(token) => Some(token),
                _ => None,
            },
            _ => None,
        }
    }

    /// True if this variable can be matched on, ie. it
    /// is both a Variable node and is not a VariableKind::TypeConstructor
    fn is_matchable_variable(&self) -> bool {
        match self {
            Ast::Variable(variable) => !matches!(variable.kind, VariableKind::TypeConstructor(..)),
            _ => false,
        }
    }

    pub fn integer(x: u64, kind: Option<IntegerKind>, location: &Location) -> Ast {
        Ast::Literal(Literal { kind: LiteralKind::Integer(x, kind), location: location.clone(), typ: None })
    }

    pub fn float(x: f64, kind: Option<FloatKind>, location: &Location) -> Ast {
        Ast::Literal(Literal { kind: LiteralKind::Float(x.to_bits(), kind), location: location.clone(), typ: None })
    }

    pub fn string(x: String, location: &Location) -> Ast {
        Ast::Literal(Literal { kind: LiteralKind::String(x), location: location.clone(), typ: None })
    }

    pub fn char_literal(x: char, location: &Location) -> Ast {
        Ast::Literal(Literal { kind: LiteralKind::Char(x), location: location.clone(), typ: None })
    }

    pub fn bool_literal(x: bool, location: &Location) -> Ast {
        Ast::Literal(Literal { kind: LiteralKind::Bool(x), location: location.clone(), typ: None })
    }

    pub fn unit_literal(location: &Location) -> Ast {
        Ast::Literal(Literal { kind: LiteralKind::Unit, location: location.clone(), typ: None })
    }

    pub fn variable(module_prefix: Vec<String>, name: String, location: &Location) -> Ast {
        Ast::Variable(Variable {
            kind: VariableKind::Identifier(name),
            module_prefix,
            location: location.clone(),
            definition: None,
            id: None,
            impl_scope: None,
            instantiation_mapping: Rc::new(HashMap::new()),
            typ: None,
        })
    }

    pub fn operator(operator: Token, location: &Location) -> Ast {
        Ast::Variable(Variable {
            kind: VariableKind::Operator(operator),
            module_prefix: vec![],
            location: location.clone(),
            definition: None,
            id: None,
            impl_scope: None,
            instantiation_mapping: Rc::new(HashMap::new()),
            typ: None,
        })
    }

    pub fn type_constructor(module_prefix: Vec<String>, name: String, location: &Location) -> Ast {
        Ast::Variable(Variable {
            kind: VariableKind::TypeConstructor(name),
            location: location.clone(),
            module_prefix,
            definition: None,
            id: None,
            impl_scope: None,
            instantiation_mapping: Rc::new(HashMap::new()),
            typ: None,
        })
    }

    pub fn lambda(args: Vec<Ast>, return_type: Option<Type>, body: Ast, location: &Location) -> Ast {
        assert!(!args.is_empty());
        Ast::Lambda(Lambda {
            args,
            body: Box::new(body),
            closure_environment: BTreeMap::new(),
            return_type,
            location: location.clone(),
            required_traits: vec![],
            typ: None,
        })
    }

    pub fn function_call(function: Ast, args: Vec<Ast>, location: &Location) -> Ast {
        assert!(!args.is_empty());
        Ast::FunctionCall(FunctionCall { function: Box::new(function), args, location: location.clone(), typ: None })
    }

    pub fn if_expr(condition: Ast, then: Ast, otherwise: Option<Ast>, location: &Location) -> Ast {
        if let Some(otherwise) = otherwise {
            Ast::If(If {
                condition: Box::new(condition),
                then: Box::new(then),
                otherwise: Box::new(otherwise),
                location: location.clone(),
                typ: None,
            })
        } else {
            super::desugar::desugar_if_with_no_else(condition, then, location)
        }
    }

    pub fn definition(pattern: Ast, expr: Ast, location: &Location) -> Ast {
        Ast::Definition(Definition {
            pattern: Box::new(pattern),
            expr: Box::new(expr),
            location: location.clone(),
            mutable: false,
            level: None,
            info: None,
            typ: None,
        })
    }

    pub fn match_expr(expression: Ast, mut branches: Vec<(Ast, Ast)>, location: &Location) -> Ast {
        // (Issue #80) When compiling a match statement with a single variable branch e.g:
        // `match ... | x -> ... ` a single Leaf node will be emitted as the decision tree
        // after type checking which causes us to fail since `x` will not be bound to anything
        // without a `Case` node being present. This is a hack to avoid this situation by compiling
        // this class of expressions into let bindings instead.
        if branches.len() == 1 && branches[0].0.is_matchable_variable() {
            let (pattern, rest) = branches.pop().unwrap();
            let definition = Ast::definition(pattern, expression, location);
            // TODO: turning this into a sequence can leak names in the match branch to surrounding
            // code. Soundness-wise this isn't an issue since in this case we know it will always
            // match, but it is an inconsistency that should be fixed.
            Ast::sequence(vec![definition, rest], location)
        } else {
            Ast::Match(Match {
                expression: Box::new(expression),
                branches,
                decision_tree: None,
                location: location.clone(),
                typ: None,
            })
        }
    }

    pub fn type_definition(
        name: String, args: Vec<String>, definition: TypeDefinitionBody, location: &Location,
    ) -> Ast {
        Ast::TypeDefinition(TypeDefinition {
            name,
            args,
            definition,
            location: location.clone(),
            type_info: None,
            typ: None,
        })
    }

    pub fn type_annotation(lhs: Ast, rhs: Type, location: &Location) -> Ast {
        Ast::TypeAnnotation(TypeAnnotation { lhs: Box::new(lhs), rhs, location: location.clone(), typ: None })
    }

    pub fn import(path: Vec<String>, location: &Location, symbols: HashSet<String>) -> Ast {
        assert!(!path.is_empty());
        Ast::Import(Import { path, location: location.clone(), typ: None, module_id: None, symbols })
    }

    pub fn trait_definition(
        name: String, args: Vec<String>, fundeps: Vec<String>, declarations: Vec<TypeAnnotation>, location: &Location,
    ) -> Ast {
        assert!(!args.is_empty());
        Ast::TraitDefinition(TraitDefinition {
            name,
            args,
            fundeps,
            declarations,
            location: location.clone(),
            level: None,
            trait_info: None,
            typ: None,
        })
    }

    pub fn trait_impl(
        trait_name: String, trait_args: Vec<Type>, given: Vec<Trait>, definitions: Vec<Definition>, location: &Location,
    ) -> Ast {
        assert!(!trait_args.is_empty());
        Ast::TraitImpl(TraitImpl {
            trait_name,
            trait_args,
            given,
            definitions,
            location: location.clone(),
            trait_arg_types: vec![],
            impl_id: None,
            trait_info: None,
            typ: None,
        })
    }

    pub fn return_expr(expression: Ast, location: &Location) -> Ast {
        Ast::Return(Return { expression: Box::new(expression), location: location.clone(), typ: None })
    }

    pub fn sequence(statements: Vec<Ast>, location: &Location) -> Ast {
        assert!(!statements.is_empty());
        Ast::Sequence(Sequence { statements, location: location.clone(), typ: None })
    }

    pub fn extern_expr(declarations: Vec<TypeAnnotation>, location: &Location) -> Ast {
        Ast::Extern(Extern { declarations, location: location.clone(), level: None, typ: None })
    }

    pub fn member_access(lhs: Ast, field: String, is_offset: bool, location: &Location) -> Ast {
        Ast::MemberAccess(MemberAccess { lhs: Box::new(lhs), field, is_offset, location: location.clone(), typ: None })
    }

    pub fn assignment(lhs: Ast, rhs: Ast, location: &Location) -> Ast {
        Ast::Assignment(Assignment { lhs: Box::new(lhs), rhs: Box::new(rhs), location: location.clone(), typ: None })
    }

    pub fn effect_definition(
        name: String, args: Vec<String>, declarations: Vec<TypeAnnotation>, location: &Location,
    ) -> Ast {
        Ast::EffectDefinition(EffectDefinition {
            name,
            args,
            declarations,
            location: location.clone(),
            level: None,
            typ: None,
            effect_info: None,
        })
    }

    pub fn handle(expression: Ast, branches: Vec<(Ast, Ast)>, location: &Location) -> Ast {
        let branches = super::desugar::desugar_handle_branches_into_matches(branches);
        Ast::Handle(Handle {
            expression: Box::new(expression),
            branches,
            location: location.clone(),
            resumes: vec![],
            typ: None,
        })
    }

    pub fn named_constructor(constructor: Ast, args: Vec<(String, Ast)>, location: &Location) -> Ast {
        Ast::NamedConstructor(NamedConstructor {
            constructor: Box::new(constructor),
            args,
            location: location.clone(),
            typ: None,
        })
    }

    /// This is a bit of a hack.
    /// Create a new 'scope' by wrapping body in `match () | () -> body`
    pub fn new_scope(body: Ast, location: &Location) -> Ast {
        Ast::match_expr(Ast::unit_literal(location), vec![(Ast::unit_literal(location), body)], location)
    }
}

/// A macro for calling a method on every variant of an Ast node.
/// Useful for implementing a trait for the Ast and every node inside.
/// This is used for all compiler passes, as well as the Locatable trait below.
macro_rules! dispatch_on_expr {
    ( $expr_name:expr, $function:expr $(, $($args:expr),* )? ) => ({
        match $expr_name {
            $crate::parser::ast::Ast::Literal(inner) =>          $function(inner $(, $($args),* )? ),
            $crate::parser::ast::Ast::Variable(inner) =>         $function(inner $(, $($args),* )? ),
            $crate::parser::ast::Ast::Lambda(inner) =>           $function(inner $(, $($args),* )? ),
            $crate::parser::ast::Ast::FunctionCall(inner) =>     $function(inner $(, $($args),* )? ),
            $crate::parser::ast::Ast::Definition(inner) =>       $function(inner $(, $($args),* )? ),
            $crate::parser::ast::Ast::If(inner) =>               $function(inner $(, $($args),* )? ),
            $crate::parser::ast::Ast::Match(inner) =>            $function(inner $(, $($args),* )? ),
            $crate::parser::ast::Ast::TypeDefinition(inner) =>   $function(inner $(, $($args),* )? ),
            $crate::parser::ast::Ast::TypeAnnotation(inner) =>   $function(inner $(, $($args),* )? ),
            $crate::parser::ast::Ast::Import(inner) =>           $function(inner $(, $($args),* )? ),
            $crate::parser::ast::Ast::TraitDefinition(inner) =>  $function(inner $(, $($args),* )? ),
            $crate::parser::ast::Ast::TraitImpl(inner) =>        $function(inner $(, $($args),* )? ),
            $crate::parser::ast::Ast::Return(inner) =>           $function(inner $(, $($args),* )? ),
            $crate::parser::ast::Ast::Sequence(inner) =>         $function(inner $(, $($args),* )? ),
            $crate::parser::ast::Ast::Extern(inner) =>           $function(inner $(, $($args),* )? ),
            $crate::parser::ast::Ast::MemberAccess(inner) =>     $function(inner $(, $($args),* )? ),
            $crate::parser::ast::Ast::Assignment(inner) =>       $function(inner $(, $($args),* )? ),
            $crate::parser::ast::Ast::EffectDefinition(inner) => $function(inner $(, $($args),* )? ),
            $crate::parser::ast::Ast::Handle(inner) =>           $function(inner $(, $($args),* )? ),
            $crate::parser::ast::Ast::NamedConstructor(inner) => $function(inner $(, $($args),* )? ),
        }
    });
}

impl Locatable for Ast {
    fn locate(&self) -> Location {
        dispatch_on_expr!(self, Locatable::locate)
    }
}

macro_rules! impl_locatable_for {
    ( $name:tt ) => {
        impl Locatable for $name {
            fn locate(&self) -> Location {
                self.location.clone()
            }
        }
    };
}

impl_locatable_for!(Literal);
impl_locatable_for!(Variable);
impl_locatable_for!(Lambda);
impl_locatable_for!(FunctionCall);
impl_locatable_for!(Definition);
impl_locatable_for!(If);
impl_locatable_for!(Match);
impl_locatable_for!(TypeDefinition);
impl_locatable_for!(TypeAnnotation);
impl_locatable_for!(Import);
impl_locatable_for!(TraitDefinition);
impl_locatable_for!(TraitImpl);
impl_locatable_for!(Return);
impl_locatable_for!(Sequence);
impl_locatable_for!(Extern);
impl_locatable_for!(MemberAccess);
impl_locatable_for!(Assignment);
impl_locatable_for!(EffectDefinition);
impl_locatable_for!(Handle);
impl_locatable_for!(NamedConstructor);

impl Locatable for Type {
    fn locate(&self) -> Location {
        match self {
            Type::Integer(_, location) => location.clone(),
            Type::Float(_, location) => location.clone(),
            Type::Char(location) => location.clone(),
            Type::String(location) => location.clone(),
            Type::Pointer(location) => location.clone(),
            Type::Boolean(location) => location.clone(),
            Type::Unit(location) => location.clone(),
            Type::Reference(location) => location.clone(),
            Type::Function(_, _, _, _, location) => location.clone(),
            Type::TypeVariable(_, location) => location.clone(),
            Type::UserDefined(_, location) => location.clone(),
            Type::TypeApplication(_, _, location) => location.clone(),
            Type::Pair(_, _, location) => location.clone(),
        }
    }
}
