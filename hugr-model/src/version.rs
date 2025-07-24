use derive_more::derive::Display;
use std::str::FromStr;
use thiserror::Error;

/// A version number.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Display)]
#[display("{major}.{minor}")]
pub struct Version {
    /// The major part of the version.
    pub major: u32,
    /// The minor part of the version.
    pub minor: u32,
}

impl FromStr for Version {
    type Err = VersionParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (major, minor) = s.split_once(".").ok_or(VersionParseError)?;
        let major = major.parse().map_err(|_| VersionParseError)?;
        let minor = minor.parse().map_err(|_| VersionParseError)?;
        Ok(Self { major, minor })
    }
}

/// Error when parsing a [`Version`].
#[derive(Debug, Clone, Error)]
#[error("failed to parse version")]
pub struct VersionParseError;

#[cfg(test)]
mod test {
    use super::Version;
    use std::str::FromStr;

    #[test]
    fn test_parse() {
        assert_eq!(
            Version::from_str("0.1").unwrap(),
            Version { major: 0, minor: 1 }
        );
        assert_eq!(
            Version::from_str("1337.0").unwrap(),
            Version {
                major: 1337,
                minor: 0
            }
        );
        assert!(Version::from_str("0").is_err());
        assert!(Version::from_str("0.").is_err());
        assert!(Version::from_str("0.1.").is_err());
        assert!(Version::from_str("0...").is_err());
        assert!(Version::from_str("").is_err());
    }
}
