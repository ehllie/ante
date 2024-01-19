//! location.rs - Defines the Location struct for storing source
//! Locations throughout the compiler. Most notably, these locations
//! are passed around throughout the parser and are stored in each
//! Ast node, along with several structs in the ModuleCache.
use std::{path::Path, sync::Arc};

/// A given Position in a file. These are usually used as
/// start positions for a Location struct.
///
/// TODO: remove line and column fields to make Position smaller
/// and faster. These can be computed on demand while issuing
/// error messages. Since Locations are used pervasively in the
/// lexer and parser, this would likely speed up compilation.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Position {
    pub index: usize,
    pub line: u32,
    pub column: u16,
}

impl Position {
    /// The first position in a file
    pub fn begin() -> Position {
        Position { index: 0, line: 1, column: 1 }
    }

    /// Increment the position 1 character forward
    pub fn advance(&mut self, char_len: usize, passed_newline: bool) {
        if passed_newline {
            self.line += 1;
            self.column = 1;
        } else {
            self.column += 1;
        }
        self.index += char_len;
    }
}

/// An ending position. Error reporting doesn't need to report
/// the ending line/column of an error so it isn't stored here.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct EndPosition {
    pub index: usize,
}

impl EndPosition {
    pub fn new(index: usize) -> EndPosition {
        EndPosition { index }
    }
}

/// The file a Location arises from.
/// The 'c lifetime refers to the ModuleCache which stores
/// file's names and contents.
#[derive(Debug, Copy, Clone)]
pub struct File<'c> {
    pub filename: &'c Path,
    pub contents: &'c str,
}

/// A source location for a given Ast node or other construct.
/// The 'c lifetime refers to the ModuleCache which stores
/// the file paths.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Location {
    pub filename: Arc<Path>,
    pub start: Position,
    pub end: EndPosition,
}

impl Ord for Location {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.start, self.end).cmp(&(other.start, other.end))
    }
}

impl PartialOrd for Location {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Location {
    pub fn new(filename: &Path, start: Position, end: EndPosition) -> Location {
        let filename = Arc::from(filename);
        Location { filename, start, end }
    }

    /// Returns a location to an item that is built into the compiler and is not
    /// actually present in any source code. Care should be taken when defining
    /// these types to ensure errors presented to users don't point to the non-existant
    /// source location. Example of builtins are the string type and the '.' trait family.
    pub fn builtin() -> Location {
        let start = Position { index: 0, line: 0, column: 0 };
        let end = EndPosition { index: 0 };
        // TODO: update to reference prelude_path with appropriate lifetimes
        Location::new(Path::new("stdlib/prelude.an"), start, end)
    }

    pub fn length(&self) -> usize {
        self.end.index - self.start.index
    }

    /// Unify the two Locations, returning a new Location that starts at the minimum
    /// of both starting points and ends at the maximum of both end points.
    pub fn union(&self, other: &Location) -> Location {
        let start = if self.start.index < other.start.index { self.start } else { other.start };
        let end = if self.end.index < other.end.index { other.end } else { self.end };

        Location { filename: self.filename.clone(), start, end }
    }
}

/// A trait representing anything that has a Location
pub trait Locatable {
    fn locate(&self) -> Location;
}
