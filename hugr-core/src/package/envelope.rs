use std::{io::{self, BufReader, Read, Write}, mem};

use itertools::Itertools as _;
use packed_struct::prelude::{PackedStruct, PrimitiveEnum, PrimitiveEnum_u64};

use crate::{
    extension::ExtensionRegistry,
    package::{Package, PackageEncodingError, PackageError, PackageValidationError},
};

#[cfg(feature = "model_unstable")]
use crate::import::ImportError;

use std::io::BufRead;

#[derive(derive_more::Display, derive_more::Error, Debug, derive_more::From)]
pub enum EnvelopeError {
    SerdeError {
        source: serde_json::Error,
    },
    TypeNotSupported(#[error(ignore)] PayloadType),
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
}

// #[derive(packed_struct::derive::PackedStruct)]
// #[packed_struct(endian = "lsb", bit_numbering = "msb0")]
pub struct EnvelopeHeader {
    // #[packed_field(bytes = "0")]
    magic: u64,
    // #[packed_field(bytes = "8", ty = "enum")]
    payload_type: PayloadType,
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
}

pub const MAGIC_NUMBER: u64 = 0xAAAAAAAAAAAAAAAA;
pub const DEFAULT_PAYLOAD_TYPE: PayloadType = PayloadType::JsonZstd;

#[derive(PrimitiveEnum_u64, Clone, Copy, Eq, PartialEq, Debug, derive_more::Display)]
#[repr(u64)]
pub enum PayloadType {
    Json = 1,
    JsonZstd = 2,
    Model = 3,
    ModelZstd = 4,
}

impl PayloadType {
    pub fn from_str(s: impl AsRef<str>) -> Option<Self> {
        PrimitiveEnum::from_str(s.as_ref())
    }
}

pub fn read_envelope(mut reader: impl io::Read, registry: &ExtensionRegistry) -> Result<Package, EnvelopeError> {
    let header = read_header(&mut reader)?;
    decode_package(header.payload_type, reader, registry)
}

pub fn write_envelope(
    package: &Package,
    mut writer: impl io::Write,
    payload_type: Option<PayloadType>,
) -> Result<(), EnvelopeError> {
    let payload_type = payload_type.unwrap_or(DEFAULT_PAYLOAD_TYPE);
    let header = EnvelopeHeader {
        magic: MAGIC_NUMBER,
        payload_type,
    };
    write_header(&header, &mut writer)?;
    encode_package(package, payload_type, writer)
}


fn write_header(header: &EnvelopeHeader, writer: &mut impl Write) -> Result<(), EnvelopeError> {
    header.validate()?;
    let magic_bytes = header.magic.to_ne_bytes();
    let payload_type_bytes = (header.payload_type as u64).to_ne_bytes();
    writer.write_all(&magic_bytes)?;
    writer.write_all(&payload_type_bytes)?;
    Ok(())
}

fn read_header(reader: &mut impl Read) -> Result<EnvelopeHeader, EnvelopeError> {
    let mut magic_bytes = [0;8];
    let mut payload_type_bytes = [0;8];
    reader.read_exact(&mut magic_bytes)?;
    reader.read_exact(&mut payload_type_bytes)?;
    let header = EnvelopeHeader { magic: u64::from_ne_bytes(magic_bytes), payload_type: unsafe { mem::transmute(u64::from_ne_bytes(payload_type_bytes)) }};
    header.validate()?;
    Ok(header)
}

pub fn encode_package(
    package: &Package,
    payload_type: PayloadType,
    // extension_registry: &ExtensionRegistry,
    writer: impl Write,
) -> Result<(), EnvelopeError> {
    match payload_type {
        PayloadType::Json => encode_json(writer, package),
        #[cfg(feature = "zstd")]
        PayloadType::JsonZstd => {
            let mut encoder = zstd::Encoder::new(writer, 0)?;
            encode_json(&mut encoder, package)?;
            encoder.finish()?;
            Ok(())
        },
        #[cfg(feature = "model_unstable")]
        PayloadType::Model => encode_model(writer, package),
        #[cfg(all(feature = "model_unstable", feature = "zstd"))]
        PayloadType::ModelZstd => {
            let mut encoder = zstd::Encoder::new(writer, 0)?;
            encode_model(&mut encoder, package)?;
            encoder.finish()?;
            Ok(())
        },
        #[allow(unreachable_patterns)]
        unsupported => Err(EnvelopeError::TypeNotSupported(unsupported))
    }
}

fn encode_json(writer: impl Write, package: &Package) -> Result<(), EnvelopeError> {
    Ok(package.to_json_writer(writer)?)
}

fn decode_json(
    stream: impl Read,
    extension_registry: &ExtensionRegistry,
) -> Result<Package, EnvelopeError> {
    Ok(Package::from_json_reader(stream, extension_registry)?)
}

pub fn decode_package(
    payload_type: PayloadType,
    payload: impl Read,
    extension_registry: &ExtensionRegistry,
) -> Result<Package, EnvelopeError> {
    match payload_type {

        PayloadType::Json => decode_json(payload, extension_registry),
        #[cfg(feature = "zstd")]
        PayloadType::JsonZstd => decode_json(zstd::Decoder::new(payload)?, extension_registry),
        #[cfg(feature = "model_unstable")]
        PayloadType::Model =>decode_model(
                        BufReader::new(payload),
                        extension_registry,
                    ),
        #[cfg(all(feature = "model_unstable", feature = "zstd"))]
        PayloadType::ModelZstd => decode_model(
                        BufReader::new(zstd::Decoder::new(payload)?),
                        extension_registry,
                    ),
        #[allow(unreachable_patterns)]
        unsupported => Err(EnvelopeError::TypeNotSupported(unsupported))
    }
}

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
