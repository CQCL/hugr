mod binary;

use std::{fmt, io::{self, BufReader, Read, Write}, mem};

use itertools::Itertools as _;

use crate::{
    extension::ExtensionRegistry,
    package::{Package, PackageEncodingError, PackageError, PackageValidationError},
};
use packed_struct::{prelude::{packed_bits::Bits, PackedStruct, PrimitiveEnum, ReservedZeroes}, PackingError, PackingResult};

#[cfg(feature = "model_unstable")]
use crate::import::ImportError;

use std::io::BufRead;

#[derive(derive_more::Display, derive_more::Error, Debug, derive_more::From)]
pub enum EnvelopeError {
    SerdeError {
        source: serde_json::Error,
    },
    TypeNotSupported(#[error(ignore)] PayloadDescriptor),
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

#[derive(Debug, Eq, PartialEq, Clone, Copy, derive_more::Display)]
#[display("{model_or_json} zstd:{zstd} package:{is_package}")]
pub struct PayloadDescriptor {
    model_or_json: SerialisationFormat,
    zstd: bool,
    is_package: bool,
}


// impl PayloadDescriptor {
//     pub fn from_str(s: &str) -> Option<Self> {
//         <Self as PrimitiveEnum>::from_str(s)

//     }

// }

// impl PrimitiveEnum for PayloadDescriptor {
//     type Primitive = u64;

//     fn from_primitive(val: Self::Primitive) -> Option<Self> {
//         // SAFETY: Because `Self` is marked `repr(u8)`, its layout is a `repr(C)` `union`
//         // between `repr(C)` structs, each of which has the `u8` discriminant as its first
//         // field, so we can read the discriminant without offsetting the pointer.
//         // <https://doc.rust-lang.org/std/mem/fn.discriminant.html#accessing-the-numeric-value-of-the-discriminant>
//         let bytes = val.to_le_bytes();
//         let desc_bytes = bytes[1..].try_into().unwrap();
//         match bytes[0] {
//             1 => Some(PayloadDescriptor::V1(PayloadDescriptorV1::unpack(desc_bytes).ok()?)),
//             _ => None,
//         }
//     }

//     fn to_primitive(&self) -> Self::Primitive {
//         match self {
//             PayloadDescriptor::V1(_) => 1,
//         }
//     }

//     fn from_str(s: &str) -> Option<Self> {
//         todo!()
//     }

//     fn from_str_lower(s: &str) -> Option<Self> {
//         todo!()
//     }
// }

#[repr(u8)]
#[derive(Debug, Eq, PartialEq, Clone, Copy, packed_struct::derive::PrimitiveEnum, derive_more::Display)]
pub enum SerialisationFormat {
    ModelBinary = 0,
    ModelText = 1,
    Json = 2,
}

// impl From<&[u8;7]> for PayloadDescriptorV1 {
//     fn from(value: &[u8;7]) -> Self {
//         todo!()
//     }
// }


// impl PrimitiveEnum for PayloadType {
//     type Primitive = u64;

//     fn from_primitive(val: Self::Primitive) -> Option<Self> {
//         // https://doc.rust-lang.org/reference/items/enumerations.html#pointer-casting
//         let discriminant = unsafe { *(&val as *const Self::Primitive as *const u8) };
//         match discriminant {
//             1 => Some(PayloadType::V0(PayloadTypeV0::unpack_from_slice(&val.to_ne_bytes()[1..]).ok()?)),
//             _ => None,
//         }
//     }

//     fn to_primitive(&self) -> Self::Primitive {
//         todo!()
//     }

//     fn from_str(s: &str) -> Option<Self> {
//         todo!()
//     }

//     fn from_str_lower(s: &str) -> Option<Self> {
//         todo!()
//     }
// }


// impl From<PayloadDescriptor> for EnvelopeHeader {
//     fn from(desc: PayloadDescriptor) -> Self {
//         Self {  desc, magic: MAGIC_NUMBER }
//     }
// }

// impl EnvelopeHeader {
//     pub fn validate(&self) -> Result<(), EnvelopeError> {
//         if self.magic != MAGIC_NUMBER {
//             return Err(EnvelopeError::MagicNumber {
//                 expected: MAGIC_NUMBER,
//                 found: self.magic,
//             });
//         }
//         Ok(())
//     }

//     pub fn read(mut reader: impl io::Read) -> Result<Self, EnvelopeError> {
//         use packed_struct::PackedStruct;
//         let mut buf: <Self as PackedStruct>::ByteArray = Default::default();
//         reader.read_exact(&mut buf)?;
//         let envelope = Self::unpack(&buf)?;
//         envelope.validate()?;
//         Ok(envelope)
//     }

//     pub fn write(&self, mut writer: impl io::Write) -> Result<(), EnvelopeError> {
//         self.validate()?;
//         writer.write_all(&self.pack()?)?;
//         Ok(())
//     }

// }

pub const MAGIC_NUMBER: u64 = 0xAAAAAAAAAAAAAAAA;

pub enum PayloadKind {
    ModelPackage {
        json_extension_registry: bool
    },
    ModelModule,
    JsonModule,
    JsonPackage
}

// This is annoying because when you write you want to specify whether it's a
// package or a module, and whether zstd.
//
// When you read you want to take what you've got

pub struct EnvelopeHeader {
    kind: PayloadKind,
    is_zstd: bool
}
pub const ENVELOPE_HEADER_BYTES: usize = 16;

impl EnvelopeHeader {
    pub fn to_bytes(self) -> [u8; ENVELOPE_HEADER_BYTES] {
        let mut bytes = [0u8;ENVELOPE_HEADER_BYTES];
        let word1 = &mut bytes[..8].try_into().unwrap();
        *word1 = MAGIC_NUMBER.to_le_bytes();
        let word2: &mut [u8;8] = &mut bytes[8..].try_into().unwrap();
        bytes
    }

    pub fn from_bytes(bytes: [u8; ENVELOPE_HEADER_BYTES]) -> Result<Self, EnvelopeError> {
        let word1  = u64::from_le_bytes(bytes[..8].try_into().unwrap());
        if word1 != MAGIC_NUMBER {
            Err(EnvelopeError::MagicNumber {
                expected: MAGIC_NUMBER,
                found: word1
            })?
        }
        todo!()



    }

}


// pub const DEFAULT_PAYLOAD_TYPE: PayloadType = PayloadType::JsonZstd;


// impl PayloadType {
//     pub fn from_str(s: impl AsRef<str>) -> Option<Self> {
//         PrimitiveEnum::from_str(s.as_ref())
//     }
// }

// pub fn read_envelope(mut reader: impl io::Read, registry: &ExtensionRegistry) -> Result<Package, EnvelopeError> {
//     let header = EnvelopeHeader::read(&mut reader)?;
//     decode_package(header.desc, reader, registry)
// }

// pub fn write_envelope(
//     package: &Package,
//     mut writer: impl io::Write,
//     payload_type: Option<PayloadDescriptor>,
// ) -> Result<(), EnvelopeError> {
//     // let payload_type = payload_type.unwrap_or(DEFAULT_PAYLOAD_TYPE);
//     let payload_type = todo!();
//     let header = EnvelopeHeader::from(payload_type);
//     header.write(&mut writer)?;
//     encode_package(package, payload_type, writer)
// }

pub fn encode_package(
    package: &Package,
    payload_type: PayloadDescriptor,
    // extension_registry: &ExtensionRegistry,
    writer: impl Write,
) -> Result<(), EnvelopeError> {
    todo!()
    // match payload_type {
    //     PayloadType::Json => encode_json(writer, package),
    //     #[cfg(feature = "zstd")]
    //     PayloadType::JsonZstd => {
    //         let mut encoder = zstd::Encoder::new(writer, 0)?;
    //         encode_json(&mut encoder, package)?;
    //         encoder.finish()?;
    //         Ok(())
    //     },
    //     #[cfg(feature = "model_unstable")]
    //     PayloadType::Model => encode_model(writer, package),
    //     #[cfg(all(feature = "model_unstable", feature = "zstd"))]
    //     PayloadType::ModelZstd => {
    //         let mut encoder = zstd::Encoder::new(writer, 0)?;
    //         encode_model(&mut encoder, package)?;
    //         encoder.finish()?;
    //         Ok(())
    //     },
    //     #[allow(unreachable_patterns)]
    //     unsupported => Err(EnvelopeError::TypeNotSupported(unsupported))
    // }
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
    payload_type: PayloadDescriptor,
    payload: impl Read,
    extension_registry: &ExtensionRegistry,
) -> Result<Package, EnvelopeError> {
    todo!()
    // match payload_type {

    //     PayloadType::Json => decode_json(payload, extension_registry),
    //     #[cfg(feature = "zstd")]
    //     PayloadType::JsonZstd => decode_json(zstd::Decoder::new(payload)?, extension_registry),
    //     #[cfg(feature = "model_unstable")]
    //     PayloadType::Model =>decode_model(
    //                     BufReader::new(payload),
    //                     extension_registry,
    //                 ),
    //     #[cfg(all(feature = "model_unstable", feature = "zstd"))]
    //     PayloadType::ModelZstd => decode_model(
    //                     BufReader::new(zstd::Decoder::new(payload)?),
    //                     extension_registry,
    //                 ),
    //     #[allow(unreachable_patterns)]
    //     unsupported => Err(EnvelopeError::TypeNotSupported(unsupported))
    // }
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

    // use super::*;
    // #[rstest]
    // #[case(PayloadType::Json)]
    // fn round_trip_header(#[case]payload_type: PayloadDescriptor) {
    //     let header = EnvelopeHeader::from(payload_type);
    //     let mut vec: Vec<u8> = Default::default();

    //     header.write(&mut vec).unwrap();

    //     let header2 = EnvelopeHeader::read(vec.as_slice()).unwrap();

    //     assert_eq!(header, header2);
    // }
}
