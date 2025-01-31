{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, utils }:
    utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in
      {
        devShell = with pkgs; mkShell {
          buildInputs = [
            python312
            python312Packages.venvShellHook
            python312Packages.torchvision
            python312Packages.torch
            pkg-configUpstream
            (python312Packages.opencv4.override {enableGtk3 = true;})

          ];
          shellHook = ''
            python3 --version
          '';
        };
      }
    );
}
