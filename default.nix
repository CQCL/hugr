{
  rustPlatform,
  stdenv,
}:
let
  cargoToml = builtins.fromTOML (builtins.readFile ./Cargo.toml);
  src = stdenv.mkDerivation {
    name = "hugr-source";
    phases = [ "installPhase" ];
    installPhase = ''
      mkdir -p $out;
      cp ${./Cargo.lock} $out/Cargo.lock;
      cp ${./Cargo.toml} $out/Cargo.toml;
      cp -r ${./src} $out/src;
      cp -r ${./benches} $out/benches;
    '';
  };
in rustPlatform.buildRustPackage {
  inherit (cargoToml.package) name version;
  inherit src;
  cargoLock.lockFile = ./Cargo.lock;
}
