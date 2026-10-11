//! Error type.

use std::fmt;

/// Errors raised by the numerical core.
#[derive(Debug, Clone, PartialEq)]
pub enum TraceError {
    /// Array shapes do not match.
    Shape(String),
    /// An argument is outside its valid range.
    InvalidArgument(String),
    /// A linear-algebra operation failed (e.g. a matrix is not positive definite).
    Numerical(String),
}

impl fmt::Display for TraceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TraceError::Shape(s) => write!(f, "shape error: {s}"),
            TraceError::InvalidArgument(s) => write!(f, "invalid argument: {s}"),
            TraceError::Numerical(s) => write!(f, "numerical error: {s}"),
        }
    }
}

impl std::error::Error for TraceError {}

/// Result alias.
pub type Result<T> = std::result::Result<T, TraceError>;
