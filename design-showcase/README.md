# BARK! design showcase

Standalone HTML + CSS exploration for a one-page Android dog-detection app.
No JavaScript, frameworks, remote fonts, remotely hosted images, database, or model calls.

## Open

Open `C:\Work\Classes\Animal_Vocalization\design-showcase\index.html`
in Chrome or Edge. No install, build, or web server is required.

The Upload / Processing / Result radio controls switch all three concept previews
using CSS sibling selectors. They support keyboard navigation (Tab to the radio
group, then arrow keys to select a state). App upload, video, and download controls
are static visuals, not functional controls.

## Directions

- **A — Warm Companion:** cream, burnt orange, rounded friendly cards.
- **B — Clean Studio:** ivory, espresso, compact structured layouts.
- **C — Dark Kennel:** deep cocoa, warm orange, cinematic contrast.

All concepts show the same required flow: choose video, processing, and a rendered
video result with download and start-over actions. The illustrated dog frame and
its box are mock content, not evidence of model performance. The displayed filename
and duration are examples. Video format/size/duration limits are not yet finalized.

The original supplied logo is now used in all app headers, the showcase header,
the browser favicon, and an app-icon visual preview. A byte-identical copy is at
`C:\Work\Classes\Animal_Vocalization\design-showcase\assets\happy_barking_beagle_sticker.png`.
The source is `C:\Work\Classes\Animal_Vocalization\happy_barking_beagle_sticker.png`.
The transparent 1254 × 1254 PNG is displayed with contain sizing: no cropping,
recoloring, or modification to the original artwork. The CSS placeholder is removed.
Palette values remain visual suggestions rather than exact sampled colors.

## Files

- `C:\Work\Classes\Animal_Vocalization\design-showcase\index.html`
- `C:\Work\Classes\Animal_Vocalization\design-showcase\styles.css`

## Approved direction and branding

The user selected **A — Warm Companion**. Preserve its warm cream background,
burnt-orange primary actions, dark-brown typography, rounded upload card, and
spacious layout when moving to React Native.

Use the user's original supplied logo in the main-page header and as the basis
for the Android app icon. The header branding is implemented in this HTML/CSS
showcase. The cream-backed app-icon preview preserves the full artwork, but is
not an installed Android launcher icon. Native icon resources will be configured
when the React Native app is built. The other two concepts remain for reference.

The illustrated phone frame belongs only to the showcase, not the app interface.
This showcase does not change training files or load models.

## Captured previews and validation

The original logo is now larger and centered, with no additional BARK text,
tagline or badge alongside it. This is also the header used in the React Native
implementation at `C:\Work\Classes\Animal_Vocalization\mobile\App.tsx`.

Chrome-rendered screenshots are available at:

- `C:\Work\Classes\Animal_Vocalization\design-showcase\screenshots\upload-desktop.png`
- `C:\Work\Classes\Animal_Vocalization\design-showcase\screenshots\processing-desktop.png`
- `C:\Work\Classes\Animal_Vocalization\design-showcase\screenshots\result-desktop.png`
- `C:\Work\Classes\Animal_Vocalization\design-showcase\screenshots\upload-mobile.png`
- `C:\Work\Classes\Animal_Vocalization\design-showcase\screenshots\warm-upload-logo.png`
- `C:\Work\Classes\Animal_Vocalization\design-showcase\screenshots\warm-processing-logo.png`
- `C:\Work\Classes\Animal_Vocalization\design-showcase\screenshots\warm-result-logo.png`
- `C:\Work\Classes\Animal_Vocalization\design-showcase\screenshots\app-icon-logo.png`

All three states were checked at viewport widths of 320, 390, 860, 861, 1024,
and 1440 pixels. All 18 checks passed: the selected state appeared in each
concept, CSS loaded, and no page or phone-screen horizontal overflow was found.
Arrow-key selection of the native radio controls also passed. No JavaScript
runtime exceptions or remote resource requests were observed.
These checks were rerun after adding the original logo and app-icon preview.
All five logo images loaded with their original 1254-pixel intrinsic width and
contain sizing, and the copied asset hash matches the source file.

Detailed results are saved in
`C:\Work\Classes\Animal_Vocalization\design-showcase\screenshots\validation.json`.
Temporary browser automation was used only for validation; the showcase itself
contains no JavaScript. This verifies a browser design prototype, not a native
Android implementation or real inference behavior.