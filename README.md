# GaitDynamics with Shoes On!

https://github.com/user-attachments/assets/bf27ed81-0c0a-42d5-9bf3-f23b902dba55

**Live demo:** [demos.dandrea.sh/gaitdynamics-with-shoes-on](https://demos.dandrea.sh/gaitdynamics-with-shoes-on/)

Shoe-sole-shaped finite element (FE) meshes were glued to the feet of a skeletal model to demonstrate real-time deformation during walking. Given kinematic input data (walking), the "floor" height is adjusted so that contact energy is accumulated during walking as the feet strike, as the sole FE mesh intersects with the floor. 

You can guesstimate the "correct" floor offset from the feet by adjusting a slider to match the GRF recovered from the FE mesh to the GRF output produced by the GaitDynamics model for that same kinematic input. Figured this would be a fun way to demonstrate how real-time FE solver surrogates can be used without trying to create some forward simulation.

<img width="1402" height="677" alt="sole_floor_contact" src="https://github.com/user-attachments/assets/436b1488-67eb-4b01-a51a-69c9a01e9756" />

I've included a few different walking motions so the different GRF, floor offset amounts, and deformation patterns can be observed.

## Tech Stack

- **Three.js**: renders the anatomy, floor, force markers, and deforming sole meshes
- **ONNX Runtime Web**: in-browser NN inference for sole mesh deformation. Neural net trained using an NNFE approach as described by Sacks et al. [here](https://www.sciencedirect.com/science/article/abs/pii/S0045782524003165) and inspired by code [here](https://github.com/BenThomas324/NNFE).
- **Vite** static web app
- **GaitDynamics** predicts GRF/CoP from kinematic inputs when generating replay artifacts
- **OpenSim / AddBiomechanics / Nimble**: provides the standardized skeletal model and `.b3d` motion data processed to be used in webapp
- **Python + NumPy**: used only for offline artifact generation

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
