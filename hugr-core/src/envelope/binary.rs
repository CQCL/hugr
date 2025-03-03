use std::fmt;

use super::{SerialisationFormat, PayloadDescriptor};

// use deku::prelude::{DekuRead, DekuWrite};

pub const MAGIC_NUMBER: u64 = 0xAAAAAAAAAAAAAAAA;

// #[derive(Debug, Eq, PartialEq, Clone, Copy, DekuRead, DekuWrite)]
// #[deku(magic = b"HUGR0123", endian ="little", bytes = "8", id_type = "u8")]
// #[repr(u8)]
// pub enum BinaryPayloadDescriptor {
//     V1(#[deku(bytes="7")] BinaryPayloadDescriptorV1) = 1,
// }

// #[derive(Debug, Eq, PartialEq, Clone, Copy, DekuRead, DekuWrite)]
// #[deku(endian = "little")]
// struct BinaryPayloadDescriptorV1 {
//     #[deku(bytes = "1")]
//     pub format: u8,
//     #[deku(bits = "1")]
//     pub ztd: bool,
//     #[deku(bits = "1", pad_bits_after = "54")]
//     pub is_package: bool,
// }

// const PAYLOAD_DESCRIPTOR_SIZE_BYTES: usize = 7;

// impl BinaryPayloadDescriptorUnion {
//     unsafe fn pack(&self, version: BinaryPayloadDescriptorVersion) -> PackingResult<[u8;PAYLOAD_DESCRIPTOR_SIZE_BYTES]> {
//         match version {
//             BinaryPayloadDescriptorVersion::V1 => unsafe { self.v1 }.pack()
//         }
//     }

//     fn unpack(src: &[u8;PAYLOAD_DESCRIPTOR_SIZE_BYTES], version: BinaryPayloadDescriptorVersion)  -> PackingResult<Self> {
//         match version {
//             BinaryPayloadDescriptorVersion::V1 => Ok(Self {
//                 v1: PayloadDescriptorV1::unpack(src)?
//             })
//         }
//     }

//     unsafe fn eq(&self, other: Self, version: BinaryPayloadDescriptorVersion) -> bool {
//         match version {
//             BinaryPayloadDescriptorVersion::V1 => unsafe { self.v1 }.eq(&unsafe { other.v1 })
//         }
//     }

//     unsafe fn fmt(&self, f: &mut fmt::Formatter<'_>, version: BinaryPayloadDescriptorVersion) -> fmt::Result {
//         use fmt::Debug;
//         match version {
//             BinaryPayloadDescriptorVersion::V1 => unsafe { self.v1 }.fmt(f)
//         }
//     }
// }



// impl PartialEq for BinaryPayloadDescriptor {
//     fn eq(&self, other: &Self) -> bool {
//         self.version == other.version && unsafe {  self.desc.eq(other.desc, self.version) }
//     }
// }

// impl Eq for BinaryPayloadDescriptor {}

// impl fmt::Debug  for BinaryPayloadDescriptor {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         unsafe { self.desc.fmt(f, self.version) }
//     }
// }

// impl PackedStruct for BinaryPayloadDescriptor {
//     type ByteArray = [u8;PAYLOAD_DESCRIPTOR_SIZE_BYTES + 1];

//     fn pack(&self) -> PackingResult<Self::ByteArray> {
//         let mut bytes = Self::ByteArray::default();
//         bytes[0] = self.version.to_primitive();
//         bytes[1..8].copy_from_slice(&unsafe { self.desc.pack(self.version) }?);
//         Ok(bytes)
//     }

//     fn unpack(src: &Self::ByteArray) -> PackingResult<Self> {
//         let version = BinaryPayloadDescriptorVersion::from_primitive(src[0]).ok_or(PackingError::InvalidValue)?;
//         let desc = BinaryPayloadDescriptorUnion::unpack(&src[1..].try_into().unwrap(), version)?;
//         Ok(Self { version, desc })
//     }
// }
