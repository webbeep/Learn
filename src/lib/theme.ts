// Visualization colors need to know the theme (they're computed in JS, not CSS).
let dark = true;
export const setVizDark = (d: boolean) => { dark = d; };
export const isDark = () => dark;
