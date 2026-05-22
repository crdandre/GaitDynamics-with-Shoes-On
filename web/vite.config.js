// In dev, the app is served at the local Vite root.
// In production, the demo is hosted under:
//   https://demos.dandrea.sh/gaitdynamics-with-shoes-on/
//
// Override when needed:
//   VITE_BASE_PATH=/some-other-path/ npm run build
export default ({ command }) => ({
  base:
    command === "build"
      ? (process.env.VITE_BASE_PATH ?? "/gaitdynamics-with-shoes-on/")
      : "/",
  build: {
    outDir: "dist",
    assetsInlineLimit: 0,
  },
});
