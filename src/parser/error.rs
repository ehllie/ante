//! parser/error.rs - Defines the ParseError type and the formatting shown
//! when printing this error to stderr.
use super::combinators::Input;
use crate::error::location::{Locatable, Location};
use crate::error::ErrorMessage;
use crate::lexer::token::{LexerError, Token};
use crate::util::join_with;
use std::fmt::Display;

#[derive(Debug)]
pub enum ParseError {
    /// A parsing error may not be fatal if it can be ignored because
    /// e.g. the parser is currently within an `or([...])` combinator that
    /// succeeds if any of the parsers in its array succeed.
    Fatal(Box<ParseError>),

    /// Expected any of the given tokens, but found... whatever is at the
    /// source Location instead
    Expected(Vec<Token>, Location),

    /// Failed while in the given parsing rule. E.g. "failed to parse a type".
    /// Due to backtracking this error is somewhat rare since the parser tends
    /// to backtrack trying to parse something else instead of failing in the
    /// rule that parsed the furthest. Proper usage of !<- (or `no_backtracking`)
    /// helps mediate this somewhat.
    InRule(&'static str, Location),

    /// Found a Token::Invalid issued by the lexer, containing some LexerError.
    /// These errors are always wrapped in a Fatal.
    LexerError(LexerError, Location),
}

pub type ParseResult<'local, 'cache, T> = Result<(Input<'local, 'cache>, T, Location), ParseError>;

impl<'a> Locatable for ParseError {
    fn locate(&self) -> Location {
        match self {
            ParseError::Fatal(error) => error.locate(),
            ParseError::Expected(_, location) => location.clone(),
            ParseError::InRule(_, location) => location.clone(),
            ParseError::LexerError(_, location) => location.clone(),
        }
    }
}

impl<'a> Display for ParseError {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        match self {
            ParseError::Fatal(error) => error.fmt(fmt),
            ParseError::Expected(tokens, location) => {
                if tokens.len() == 1 {
                    let msg = format!("parser expected {} here", tokens[0]);
                    write!(fmt, "{}", ErrorMessage::error(&msg[..], location.clone()))
                } else {
                    let expected = join_with(tokens.iter(), ", ");
                    let msg = format!("parser expected one of {}", expected);
                    write!(fmt, "{}", ErrorMessage::error(&msg[..], location.clone()))
                }
            },
            ParseError::InRule(rule, location) => {
                let msg = format!("failed trying to parse a {}", rule);
                write!(fmt, "{}", ErrorMessage::error(&msg[..], location.clone()))
            },
            ParseError::LexerError(error, location) => {
                write!(fmt, "{}", ErrorMessage::error(&error.to_string()[..], location.clone()))
            },
        }
    }
}
