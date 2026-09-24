#!/bin/bash
# Build KreaImage.app (SwiftUI, compiled with swiftc) into the project root and sign it ad hoc.
#   app/build.sh                 -> ./KreaImage.app
#   app/build.sh --install       -> also copy it to /Applications
set -euo pipefail
cd "$(dirname "$0")"
PROJECT="$(cd .. && pwd)"
APP="$PROJECT/KreaImage.app"
BUILD="$PROJECT/app/build"
BUNDLE_ID="local.kreaimage.studio"

mkdir -p "$BUILD"
[ -f AppIcon.icns ] || "$PROJECT/.venv/bin/python" make_icon.py

echo "compiling…"
xcrun swiftc -O -parse-as-library \
  -target arm64-apple-macos14.0 \
  -sdk "$(xcrun --sdk macosx --show-sdk-path)" \
  KreaImageApp.swift -o "$BUILD/KreaImage"

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BUILD/KreaImage" "$APP/Contents/MacOS/KreaImage"
cp AppIcon.icns "$APP/Contents/Resources/AppIcon.icns"
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleIdentifier</key><string>$BUNDLE_ID</string>
  <key>CFBundleName</key><string>KreaImage</string>
  <key>CFBundleDisplayName</key><string>KreaImage</string>
  <key>CFBundleExecutable</key><string>KreaImage</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>LSApplicationCategoryType</key><string>public.app-category.graphics-design</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSPrincipalClass</key><string>NSApplication</string>
  <key>NSSupportsAutomaticTermination</key><false/>
  <key>NSSupportsSuddenTermination</key><false/>
  <!-- The UI is served by the app's own engine server on 127.0.0.1 over plain HTTP. -->
  <key>NSAppTransportSecurity</key>
  <dict><key>NSAllowsLocalNetworking</key><true/></dict>
  <!-- Default project location (override: defaults write $BUNDLE_ID projectPath /path/to/krea_metal) -->
  <key>KreaProjectPath</key><string>$PROJECT</string>
</dict>
</plist>
PLIST

echo "signing (ad hoc)…"
codesign --force --sign - --timestamp=none "$APP"
codesign --verify --strict "$APP" && echo "signature ok"

if [ "${1:-}" = "--install" ]; then
  rm -rf "/Applications/KreaImage.app"
  cp -R "$APP" "/Applications/"
  echo "installed to /Applications/KreaImage.app"
fi
echo "built $APP"
