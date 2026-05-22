# GaitDynamics with Shoes On!

**Live demo:** [demos.dandrea.sh/gaitdynamics-with-shoes-on](https://demos.dandrea.sh/gaitdynamics-with-shoes-on/)

Shoe-sole-shaped finite element (FE) meshes were glued to the feet of a skeletal model to demonstrate real-time deformation during walking. Given kinematic input data (walking), the "floor" height is adjusted so that contact energy is accumulated during walking as the feet strike, as the sole FE mesh intersects with the floor. 

You can guesstimate the "correct" floor offset from the feet (i.e., what might be considered the shoe sole thickness) by adjusting a slider to match the GRF recovered from the FE mesh to the GRF output produced by the GaitDynamics model for that same kinematic input.

I've included a few different walking motions so the different GRF, floor offset amounts, and deformation patterns can be observed.

## Run Locally

```bash
cd web
npm ci
npm run dev
```

## Working with New AddBiomechanics Files
To visualize different .b3d files from AddBiomechanics, clone [GaitDynamics](https://github.com/stanfordnmbl/GaitDynamics) in this repo folder and see `scripts/extract_b3d_kinematics.py`. This script extracts per-trial kinematics to .mot and .osim files and builds a set of files handle the motion and live NN inference quickly.

AddBiomechanics website: [https://addbiomechanics.org/](https://addbiomechanics.org/)

Google Drive link to raw source file used in demo: [`subject2.b3d`](https://drive.google.com/file/d/14ZlV97dEdm3LAYmtBkxYg7qZ_jvc6LfZ/view?usp=drive_link)

## Author and License

*Christian D'Andrea — <https://github.com/crdandre> — MIT licensed, see [LICENSE](LICENSE).*
