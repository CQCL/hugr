{ pkgs, ... }:

{
  # https://devenv.sh/basics/

  # https://devenv.sh/packages/
  # manually set rust packages rather than use devenv language support because
  # it doesn't seem to be up to date for macos yet (link error)
  packages = with pkgs; [ cargo rustc rust-analyzer rustfmt clippy ];
  env.RUST_SRC_PATH = "${pkgs.rust.packages.stable.rustPlatform.rustLibSrc}";

  # https://devenv.sh/scripts/
  scripts.hello.exec = "echo Welcome to hugr dev shell!";

  enterShell = ''
    hello
    cargo --version
  '';

  # https://devenv.sh/languages/
  # https://devenv.sh/reference/options/#languagesrustversion
  # languages.rust.enable = true;

  # https://devenv.sh/pre-commit-hooks/
  # pre-commit.hooks.shellcheck.enable = true;

  # https://devenv.sh/processes/
  # processes.ping.exec = "ping example.com";

  # See full reference at https://devenv.sh/reference/options/
}
