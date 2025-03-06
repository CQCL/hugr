use std::{io::{self, BufReader, Read, Write}, mem};

use itertools::Itertools as _;

use crate::{
    extension::ExtensionRegistry, hugr::LoadHugrError, package::{Package, PackageEncodingError, PackageError, PackageValidationError}, Hugr
};

#[cfg(feature = "model_unstable")]
use crate::import::ImportError;

use std::io::BufRead;

#[derive(derive_more::Display, derive_more::Error, Debug, derive_more::From)]
pub enum EnvelopeError {
    SerdeError {
        source: serde_json::Error,
    },
    #[display("Invalid Descriptor: {desc_bytes:?}")]
    InvalidDescriptor {
        desc_bytes: [u8;8],
    },
    TypeNotSupported(#[error(ignore)] PayloadDescriptor),
    FormatUnsupported(#[error(ignore)] PayloadFormat),
    IO {
        source: std::io::Error,
    },
    Package {
        source: PackageError,
    },
    PackageEncoding {
        source: PackageEncodingError,
    },
    PackageValidation {
        source: PackageValidationError,
    },
    #[cfg(feature = "model_unstable")]
    Import {
        source: ImportError,
    },
    #[cfg(feature = "model_unstable")]
    ModelRead {
        source: hugr_model::v0::binary::ReadError,
    },
    #[cfg(feature = "model_unstable")]
    ModelWrite {
        source: hugr_model::v0::binary::WriteError,
    },
    PackingError {
        source: packed_struct::PackingError,
    },
    #[display("Bad magic number. expected '{expected:X}' found '{found:X}'")]
    MagicNumber {
        expected: u64,
        found: u64,
    },
    LoadHugrError {
        source: LoadHugrError,
    },
    ZstdUnsupported
}

// TODO maybe we should kill this, just use PayloadDescriptor and check for magic number in read/write
pub struct EnvelopeHeader {
    magic: u64,
    desc: PayloadDescriptor,
}

impl EnvelopeHeader {
    pub fn validate(&self) -> Result<(), EnvelopeError> {
        if self.magic != MAGIC_NUMBER {
            return Err(EnvelopeError::MagicNumber {
                expected: MAGIC_NUMBER,
                found: self.magic,
            });
        }
        Ok(())
    }
    pub fn write(&self, writer: &mut impl Write) -> Result<(), EnvelopeError> {
        self.validate()?;
        writer.write_all(&self.magic.to_le_bytes())?;
        writer.write_all(&self.desc.into_bytes())?;
        Ok(())
    }

    fn read(reader: &mut impl Read) -> Result<EnvelopeHeader, EnvelopeError> {
        let mut magic_bytes = [0;8];
        let mut desc_bytes = [0;8];
        reader.read_exact(&mut magic_bytes)?;
        reader.read_exact(&mut desc_bytes)?;
        let header = Self { magic: u64::from_le_bytes(magic_bytes), desc: PayloadDescriptor::from_bytes(desc_bytes).ok_or(EnvelopeError::InvalidDescriptor { desc_bytes })? };
        header.validate()?;
        Ok(header)
    }
}

pub const MAGIC_NUMBER: u64 = 0xAAAAAAAAAAAAAAAA;

#[derive(Clone, Copy, Eq, PartialEq, Debug, derive_more::Display)]
pub enum PayloadFormat {
    // one day model will not require a json extension registry
    Model { json_extension_reg: bool },
    Json
}

impl PayloadFormat {
    pub const DEFAULT: PayloadFormat = Self::Json;

    pub fn into_bytes(self) -> [u8;4] {
        // manual, but it doesn't seem worth bringing in a bunch of machinery for this
        let version = 1;
        let (discriminant, data) = match self {
            PayloadFormat::Model { json_extension_reg } => {
                (1, [0, json_extension_reg as u8])
            },
            PayloadFormat::Json =>

            {
                (2, [0;2])
            }
        };

        [version, discriminant, data[0], data[1]]
    }

    pub fn from_bytes(bytes: [u8;4]) -> Option<Self> {
        // manual, but it doesn't seem worth bringing in a bunch of machinery for this
        let 1 = bytes[0] else {
            None?
        };
        // TODO do we care to check for zeros?
        match bytes[1] {
            1 => Some(PayloadFormat::Model { json_extension_reg: (bytes[3] & 0x1) != 0 }),
            3 => Some(PayloadFormat::Json),
            _ => None
        }
    }

    pub fn encode(
        &self,
        payload: &PackageOrHugr,
        writer: impl Write,
    ) -> Result<(), EnvelopeError> {
        match self {
            Self::Json => encode_json(writer, payload),
            #[cfg(feature = "model_unstable")]
            PayloadDescriptor::Model { json_extension_reg } => {
                encode_model(writer, payload)
            }
            #[allow(unreachable_patterns)]
            unsupported => Err(EnvelopeError::FormatUnsupported(*unsupported))
        }
    }

    pub fn decode_package(
        &self,
        reader: impl Read,
        registry: &ExtensionRegistry,
    ) -> Result<Package, EnvelopeError> {
        match self {
            Self::Json => decode_json_package(reader, registry),
            #[cfg(feature = "model_unstable")]
            PayloadDescriptor::Model { json_extension_reg } => {
                assert!(*json_extension_reg);
                decode_model(reader, registry)
            }
            #[allow(unreachable_patterns)]
            unsupported => Err(EnvelopeError::FormatUnsupported(*unsupported))
        }
    }

    pub fn decode_hugr(
        &self,
        reader: impl Read,
        registry: &ExtensionRegistry,
    ) -> Result<Hugr, EnvelopeError> {
        match self {
            Self::Json => Ok(Hugr::load_json(reader, registry)?),
            #[cfg(feature = "model_unstable")]
            PayloadDescriptor::Model { json_extension_reg } => {
                assert!(*json_extension_reg);
                let bump = hugr_model::v0::bumpalo::Bump::default();
                let module = hugr_model::v0::binary::read_from_freader(reader, &bump)?;
                Ok(import_hugr(module, registry)?)
            }
            #[allow(unreachable_patterns)]
            unsupported => Err(EnvelopeError::FormatUnsupported(*unsupported))
        }
    }
}

#[derive(Clone, Copy, Eq, PartialEq, Debug, derive_more::Display)]
#[display("payload: {format} zstd: {zstd}")]
pub struct PayloadDescriptor {
    format: PayloadFormat,
    package: bool,
    zstd: bool
}

impl PayloadDescriptor {
    pub const DEFAULT_PACKAGE: Self = Self { format: PayloadFormat::DEFAULT, zstd: if cfg!(feature = "zstd") { true } else { false }, package: true };
    pub const DEFAULT_HUGR: Self = Self {
        package: false,
        ..Self::DEFAULT_PACKAGE
    };
    pub fn into_bytes(self) -> [u8;8] {
        // manual, but it doesn't seem worth bringing in a bunch of machinery for this
        let version = 1;
        let data = [0,0,((self.package as u8) << 1) | self.zstd as u8];
        let format = self.format.into_bytes();
        [version, data[0],data[1],data[2], format[0], format[1], format[2], format[3]]
    }

    pub fn from_bytes(bytes: [u8;8]) -> Option<Self> {
        // manual, but it doesn't seem worth bringing in a bunch of machinery for this
        let 1 = bytes[0] else {
            None?
        };
        Some(Self {
            format: PayloadFormat::from_bytes([bytes[4], bytes[5], bytes[6], bytes[7]])?,
            zstd: bytes[3] & 0x1 != 0,
            package: bytes[3] & 0x2 != 0

        })
    }


    // ignores self.zstd
    fn decode_impl(&self, reader: impl Read, registry: &ExtensionRegistry) -> Result<PackageOrHugr, EnvelopeError> {
        Ok(if self.package {
            self.format.decode_package(reader, registry)?.into()
        } else {
            self.format.decode_hugr(reader, registry)?.into()
        })
    }

    pub fn decode(
        &self,
        reader: impl Read,
        registry: &ExtensionRegistry,
    ) -> Result<PackageOrHugr, EnvelopeError> {
        if self.zstd {
            #[cfg(feature = "zstd")]
            {
                self.decode_impl(zstd::Decoder::new(reader)?, registry)
            }
            #[cfg(not(feature = "zstd"))]
            {
                Err(EnvelopeError::ZstdUnsupported)
            }
        } else {
            self.decode_impl(reader, registry)
        }
    }
}

#[derive(Clone,Copy,Debug,Eq,PartialEq)]
struct ZstdConfig {
    level: i32,
}

impl Default for ZstdConfig {
    fn default() -> Self {
        Self { level: 0 } // 0 means use zstd's default
    }
}

#[derive(Clone,Copy,Debug,Eq,PartialEq)]
pub struct EnvelopeConfig {
    zstd: Option<ZstdConfig>,
    model: bool,
}

impl EnvelopeConfig {
    pub fn descriptor(&self) -> PayloadDescriptor {
        PayloadDescriptor {
            format: if self.model { PayloadFormat::Model { json_extension_reg: true } } else { PayloadFormat::Json },
            zstd: self.zstd.is_some(),
            package: true
        }
    }

    pub fn write(&self, payload: &PackageOrHugr, mut writer: impl Write) -> Result<(), EnvelopeError> {
        let desc = self.descriptor();
        let header = EnvelopeHeader {
            magic: MAGIC_NUMBER,
            desc
        };
        header.write(&mut writer)?;
        if let Some(zstd) = self.zstd {
            #[cfg(feature = "zstd")]
            {
                desc.format.encode(payload, zstd::Encoder::new(writer, zdtd.level))?
            }
            #[cfg(not(feature = "zstd"))]
            {
                Err(EnvelopeError::ZstdUnsupported)?
            }
        }
        Ok(())
    }
}

impl Default for EnvelopeConfig {
    fn default() -> Self {
        Self {
            zstd: None,
            model: false,
        }
    }
}


// TODO put this on PackageOrHugr
fn encode_json(writer: impl Write, payload: &PackageOrHugr) -> Result<(), EnvelopeError> {
    match payload {
        PackageOrHugr::Package(pkg) => serde_json::to_writer(writer, pkg)?,
        PackageOrHugr::Hugr(hugr) => serde_json::to_writer(writer, hugr)?
    }
    Ok(())
}

fn decode_json_package(
    stream: impl Read,
    extension_registry: &ExtensionRegistry,
) -> Result<Package, EnvelopeError> {
    Ok(Package::from_json_reader(stream, extension_registry)?)
}

// pub fn decode_package(
//     payload_type: PayloadFormat,
//     payload: impl Read,
//     extension_registry: &ExtensionRegistry,
// ) -> Result<Package, EnvelopeError> {

//     match payload_type {
//         PayloadDesriptor::Json => decode_json(payload, extension_registry),
//         #[cfg(feature = "zstd")]
//         PayloadDesriptor::JsonZstd => decode_json(zstd::Decoder::new(payload)?, extension_registry),
//         #[cfg(feature = "model_unstable")]
//         PayloadDesriptor::Model =>decode_model(
//                         BufReader::new(payload),
//                         extension_registry,
//                     ),
//         #[cfg(all(feature = "model_unstable", feature = "zstd"))]
//         PayloadDesriptor::ModelZstd => decode_model(
//                         BufReader::new(zstd::Decoder::new(payload)?),
//                         extension_registry,
//                     ),
//         #[allow(unreachable_patterns)]
//         unsupported => Err(EnvelopeError::TypeNotSupported(unsupported))
//     }
// }

#[cfg(feature = "model_unstable")]
fn encode_model(mut writer: impl Write, package: &Package) -> Result<(), EnvelopeError> {
    use crate::export::export_hugr_list;
    use hugr_model::v0::{binary::write_module_list_to_writer, bumpalo::Bump};
    let bump = Bump::default();
    write_module_list_to_writer(&export_hugr_list(package.as_ref(), &bump), &mut writer)?;
    serde_json::to_writer(writer, &package.extensions.iter().collect_vec())?;
    Ok(())
}

#[cfg(feature = "model_unstable")]
fn decode_model(
    mut stream: impl BufRead,
    extension_registry: &ExtensionRegistry,
) -> Result<Package, EnvelopeError> {
    // TODO we don't use extension_registry
    use hugr_model::v0::{binary::read_module_list_from_reader, bumpalo::Bump};
    let bump = Bump::default();
    use crate::{import::import_hugr_list, Extension};
    let module_list = read_module_list_from_reader(&mut stream, &bump)?;
    let extensions = ExtensionRegistry::new(serde_json::from_reader::<_,Vec<Extension>>(stream)?.into_iter().map_into());
    let modules = import_hugr_list(&module_list, &extensions)?;
    let package = Package { modules, extensions  };
    package.validate()?;
    Ok(package)
}

/// A simple enum containing either a package or a single hugr.
///
/// This is required since `Package`s can only contain module-rooted hugrs.
#[derive(Debug, Clone, PartialEq, derive_more::From)]
pub enum PackageOrHugr {
    /// A package with module-rooted HUGRs and some required extensions.
    Package(Package),
    /// An arbitrary HUGR.
    Hugr(Hugr),
}

impl PackageOrHugr {
    /// Returns the list of hugrs in the package.
    pub fn into_hugrs(self) -> Vec<Hugr> {
        match self {
            PackageOrHugr::Package(pkg) => pkg.modules,
            PackageOrHugr::Hugr(hugr) => vec![hugr],
        }
    }

    /// Validates the package or hugr.
    pub fn validate(&self) -> Result<(), PackageValidationError> {
        match self {
            PackageOrHugr::Package(pkg) => pkg.validate(),
            PackageOrHugr::Hugr(hugr) => Ok(hugr.validate()?),
        }
    }
}

impl AsRef<[Hugr]> for PackageOrHugr {
    fn as_ref(&self) -> &[Hugr] {
        match self {
            PackageOrHugr::Package(pkg) => &pkg.modules,
            PackageOrHugr::Hugr(hugr) => std::slice::from_ref(hugr),
        }
    }
}
