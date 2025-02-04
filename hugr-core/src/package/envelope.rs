use std::{io::{self, BufReader, Read, Write}, mem};

use itertools::Itertools as _;
use packed_struct::{prelude::{PackedStruct, PrimitiveEnum, PrimitiveEnum_u64}, types::ReservedZeroes, PackedStructSlice};

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

#[derive(Debug, Eq, PartialEq)]
#[derive(packed_struct::derive::PackedStruct)]
#[packed_struct(endian = "lsb", bit_numbering = "msb0", size_bytes=16)]
pub struct EnvelopeHeader {
    #[packed_field(bytes = "0..", size_bytes = "8")]
    magic: u64,
    #[packed_field(bytes = "8..", size_bytes="8", ty = "enum")]
    payload_type: PayloadType,
}


#[derive(Debug, Eq, PartialEq, Clone, Copy)]
#[derive(packed_struct::derive::PrimitiveEnum_u8)]
#[repr(u8)]
pub enum ModelOrJson {
    Model = 0,
    Json = 1,
}


#[derive(Debug, Eq, PartialEq, Clone,Copy)]
#[derive(packed_struct::derive::PackedStruct)]
#[packed_struct(endian = "lsb", bit_numbering = "msb0", size_bytes=7)]
pub struct PayloadTypeV0 {
    #[packed_field(bits = "0..", size_bits="51")]
    _reserved: ReservedZeroes::<packed_struct::prelude::packed_bits::Bits::<51>>,
    #[packed_field(bits = "51", size_bits="1", ty="enum")]
    model_or_json: ModelOrJson,
    #[packed_field(bits = "52", size_bits="1")]
    zstd: bool,
    #[packed_field(bits = "53", size_bits="1")]
    is_package: bool,
}

#[derive(Debug,Eq,PartialEq,Copy,Clone)]
#[repr(u8)]
pub enum PayloadType {
    V0(PayloadTypeV0) = 1
}

impl PrimitiveEnum for PayloadType {
    type Primitive = u64;

    fn from_primitive(val: Self::Primitive) -> Option<Self> {
        // https://doc.rust-lang.org/reference/items/enumerations.html#pointer-casting
        let discriminant = unsafe { *(&val as *const Self::Primitive as *const u8) };
        match discriminant {
            1 => Some(PayloadType::V0(PayloadTypeV0::unpack_from_slice(&val.to_ne_bytes()[1..]).ok()?)),
            _ => None,
        }
    }

    fn to_primitive(&self) -> Self::Primitive {
        todo!()
    }

    fn from_str(s: &str) -> Option<Self> {
        todo!()
    }

    fn from_str_lower(s: &str) -> Option<Self> {
        todo!()
    }
}


impl From<PayloadType> for EnvelopeHeader {
    fn from(payload_type: PayloadType) -> Self {
        Self {  payload_type, magic: MAGIC_NUMBER }
    }
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

    pub fn read(mut reader: impl io::Read) -> Result<Self, EnvelopeError> {
        use packed_struct::PackedStruct;
        let mut buf: <Self as PackedStruct>::ByteArray = Default::default();
        reader.read_exact(&mut buf)?;
        let envelope = Self::unpack(&buf)?;
        envelope.validate()?;
        Ok(envelope)
    }

    pub fn write(&self, mut writer: impl io::Write) -> Result<(), EnvelopeError> {
        self.validate()?;
        writer.write_all(&self.pack()?)?;
        Ok(())
    }

}

pub const MAGIC_NUMBER: u64 = 0xAAAAAAAAAAAAAAAA;
pub const DEFAULT_PAYLOAD_TYPE: PayloadType = PayloadType::JsonZstd;


impl PayloadType {
    pub fn from_str(s: impl AsRef<str>) -> Option<Self> {
        PrimitiveEnum::from_str(s.as_ref())
    }
}

pub fn read_envelope(mut reader: impl io::Read, registry: &ExtensionRegistry) -> Result<Package, EnvelopeError> {
    let header = EnvelopeHeader::read(&mut reader)?;
    decode_package(header.payload_type, reader, registry)
}

pub fn write_envelope(
    package: &Package,
    mut writer: impl io::Write,
    payload_type: Option<PayloadType>,
) -> Result<(), EnvelopeError> {
    let payload_type = payload_type.unwrap_or(DEFAULT_PAYLOAD_TYPE);
    let header = EnvelopeHeader::from(payload_type);
    header.write(&mut writer)?;
    encode_package(package, payload_type, writer)
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

#[cfg(test)]
mod test {
    use rstest::rstest;

    use super::*;
    #[rstest]
    #[case(PayloadType::Json)]
    fn round_trip_header(#[case]payload_type: PayloadType) {
        let header = EnvelopeHeader::from(payload_type);
        let mut vec: Vec<u8> = Default::default();

        header.write(&mut vec).unwrap();

        let header2 = EnvelopeHeader::read(vec.as_slice()).unwrap();

        assert_eq!(header, header2);
    }
}
