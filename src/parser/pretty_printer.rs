//! Defines a simple pretty printer to print the Ast to stdout.
//! Used for the golden tests testing parsing to ensure there
//! are no parsing regressions.
use crate::parser::ast::{self, Ast};
use crate::util::{fmap, join_with};
use std::fmt::{self, Display, Formatter};
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

static INDENT_LEVEL: AtomicUsize = AtomicUsize::new(0);

impl<'a> Display for Ast {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        dispatch_on_expr!(self, Display::fmt, f)
    }
}

impl<'a> Display for ast::Literal {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        use ast::LiteralKind::*;
        match &self.kind {
            Integer(x, _) => write!(f, "{}", x),
            Float(x, _) => write!(f, "{}", f64::from_bits(*x)),
            String(s) => write!(f, "\"{}\"", s),
            Char(c) => write!(f, "'{}'", c),
            Bool(b) => write!(f, "{}", if *b { "true" } else { "false" }),
            Unit => write!(f, "()"),
        }
    }
}

impl<'a> Display for ast::Variable {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        use ast::VariableKind::*;
        let mut prefix = self.module_prefix.join(".");
        if !prefix.is_empty() {
            prefix += ".";
        }
        match &self.kind {
            Identifier(name) => write!(f, "{}{}", prefix, name),
            Operator(token) => write!(f, "{}", token),
            TypeConstructor(name) => write!(f, "{}{}", prefix, name),
        }
    }
}

impl<'a> Display for ast::Lambda {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "(fn")?;
        for arg in self.args.iter() {
            write!(f, " {}", arg)?;
        }
        if let Some(typ) = &self.return_type {
            write!(f, " : {}", typ)?;
        }
        write!(f, " -> {})", self.body)
    }
}

impl<'a> Display for ast::FunctionCall {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "({} {})", self.function, join_with(&self.args, " "))
    }
}

impl<'a> Display for ast::Definition {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "({} = {})", self.pattern, self.expr)
    }
}

impl<'a> Display for ast::If {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "(if {} then {} else {})", self.condition, self.then, self.otherwise)
    }
}

impl<'a> Display for ast::Match {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "(match {}", self.expression)?;
        for (pattern, branch) in self.branches.iter() {
            write!(f, " ({} {})", pattern, branch)?;
        }
        write!(f, ")")
    }
}

impl<'a> Display for ast::Type {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        use ast::Type::*;
        match self {
            Integer(Some(kind), _) => write!(f, "{}", kind),
            Integer(None, _) => write!(f, "Int"),
            Float(Some(kind), _) => write!(f, "{}", kind),
            Float(None, _) => write!(f, "Float"),
            Char(_) => write!(f, "Char"),
            String(_) => write!(f, "String"),
            Pointer(_) => write!(f, "Ptr"),
            Boolean(_) => write!(f, "Bool"),
            Unit(_) => write!(f, "Unit"),
            Reference(_) => write!(f, "Ref"),
            TypeVariable(name, _) => write!(f, "{}", name),
            UserDefined(name, _) => write!(f, "{}", name),
            Function(params, return_type, varargs, is_closure, _) => {
                let arrow = if *is_closure { "=>" } else { "->" };
                let varargs = if *varargs { "... " } else { "" };
                write!(f, "({} {}{} {})", join_with(params, " "), varargs, arrow, return_type)
            },
            TypeApplication(constructor, args, _) => {
                write!(f, "({} {})", constructor, join_with(args, " "))
            },
            Pair(first, rest, _) => {
                write!(f, "({}, {})", first, rest)
            },
        }
    }
}

impl<'a> Display for ast::TypeDefinitionBody {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        use ast::TypeDefinitionBody::*;
        match self {
            Union(types) => {
                for (name, variant_fields, _) in types {
                    let s = join_with(variant_fields, " ");
                    write!(f, "| {} {}", name, s)?;
                }
                Ok(())
            },
            Struct(types) => {
                let types = fmap(types, |(name, ty, _)| format!("{}: {}", name, ty));
                write!(f, "{}", types.join(", "))
            },
            Alias(alias) => write!(f, "{}", alias),
        }
    }
}

impl<'a> Display for ast::TypeDefinition {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let args = join_with(&self.args, "");
        write!(f, "(type {} {} = {})", self.name, args, self.definition)
    }
}

impl<'a> Display for ast::TypeAnnotation {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "(: {} {})", self.lhs, self.rhs)
    }
}

impl<'a> Display for ast::Import {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut import_path = join_with(&self.path, ".");
        let symbols = join_with(&self.symbols, " ");
        if !symbols.is_empty() {
            import_path.push('.');
        }
        write!(f, "(import {}{})", import_path, symbols)
    }
}

impl<'a> Display for ast::TraitDefinition {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "(trait {} {} ", self.name, join_with(&self.args, " "))?;
        if !self.fundeps.is_empty() {
            write!(f, "-> {} ", join_with(&self.fundeps, " "))?;
        }
        write!(f, "with\n    {}\n)", join_with(&self.declarations, "\n    "))
    }
}

impl<'a> Display for ast::TraitImpl {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let args = join_with(&self.trait_args, " ");
        let definitions = join_with(&self.definitions, "\n    ");
        let given = join_with(&self.given, " ");
        write!(
            f,
            "(impl {} {}{}{} with\n    {}\n)",
            self.trait_name,
            args,
            if !given.is_empty() { " given " } else { "" },
            given,
            definitions
        )
    }
}

impl<'a> Display for ast::Trait {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let args = join_with(&self.args, " ");
        write!(f, "({} {})", self.name, args)
    }
}

impl<'a> Display for ast::Return {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "(return {})", self.expression)
    }
}

impl<'a> Display for ast::Sequence {
    /// Whenever printing out a Sequence, pretty-print the indented
    /// block as well so that larger programs are easier to read.
    ///
    /// To do this, each Sequence prepends 4 spaces to each line of
    /// the string form of its statements unless this is the top-level
    /// Sequence, in which case we don't want any spaces before the
    /// top-level definitions.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut statements = String::new();
        let indent_level = INDENT_LEVEL.fetch_add(1, Ordering::SeqCst);

        for (i, statement) in self.statements.iter().enumerate() {
            let statement = statement.to_string();

            for line in statement.lines() {
                statements += "\n";
                if indent_level != 0 {
                    statements += "    ";
                }
                statements += line;
            }

            if i != self.statements.len() - 1 {
                statements += ";"
            }
        }

        INDENT_LEVEL.fetch_sub(1, Ordering::SeqCst);
        statements += "\n";
        write!(f, "{}", statements)?;
        Ok(())
    }
}

impl<'a> Display for ast::Extern {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "(extern\n    {})", join_with(&self.declarations, "\n    "))
    }
}

impl<'a> Display for ast::MemberAccess {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "({}.{})", self.lhs, self.field)
    }
}

impl<'a> Display for ast::Assignment {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "({} := {})", self.lhs, self.rhs)
    }
}

impl<'a> Display for ast::EffectDefinition {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "(effect {} ", self.name)?;
        if !self.args.is_empty() {
            write!(f, "{} ", join_with(&self.args, " "))?;
        }
        write!(f, "with\n    {}\n)", join_with(&self.declarations, "\n    "))
    }
}

impl<'a> Display for ast::Handle {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "(handle {}", self.expression)?;
        for (pattern, branch) in self.branches.iter() {
            write!(f, " ({} {})", pattern, branch)?;
        }
        write!(f, ")")
    }
}

impl<'a> Display for ast::NamedConstructor {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let args = fmap(self.args.iter(), |(name, expr)| format!("{name} = {expr}"));
        write!(f, "({} with {})", self.constructor, args.join(", "))
    }
}
