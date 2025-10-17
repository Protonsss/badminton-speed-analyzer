# SCSS Structure

## Setup

Install dependencies:
```bash
npm install
```

## Development

Watch for changes and compile SCSS:
```bash
npm run scss:watch
```

Compile once:
```bash
npm run scss
```

Production build (minified):
```bash
npm run scss:prod
```

## File Structure

```
scss/
├── _variables.scss    # Design tokens (colors, spacing, typography)
├── _base.scss         # Reset & base styles
├── _header.scss       # Header component
├── _hero.scss         # Hero section
├── _layout.scss       # Grid layout
├── _card.scss         # Card component
├── _buttons.scss      # Button styles
├── _forms.scss        # Form controls
├── _media.scss        # Video & canvas
├── _results.scss      # Results display
├── _modal.scss        # Modal component
├── _footer.scss       # Footer
└── main.scss          # Main entry (imports all)
```

## Usage

The SCSS is compiled to `styles.css` which is referenced in `index.html`.

All design tokens (colors, spacing, etc.) are in `_variables.scss` for easy theming.

