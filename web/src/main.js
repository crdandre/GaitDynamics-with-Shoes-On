import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import * as ort from "onnxruntime-web/wasm";
import ortWasmMjsUrl from "./vendor/onnxruntime-web/ort-wasm-simd-threaded.mjs?url";
import ortWasmUrl from "./vendor/onnxruntime-web/ort-wasm-simd-threaded.wasm?url";
import "./styles.css";

const BASE_PATH = import.meta.env.BASE_URL;
const MOTION_ARTIFACTS = [
  {
    id: "opensim-walk",
    label: "OpenSim Walking",
    root: `${BASE_PATH}artifacts/gait_nnfe_replay`,
    defaultFloorY: -0.038,
    description:
      "OpenSim 4.5 walking kinematics: coordinates.mot + subject_walk_armless_18musc.osim, with GaitDynamics-predicted GRF/CoP.",
  },
  {
    id: "wang2023-subj10-walking1",
    label: "No-Arms Subject 10, Wang 2023: walking1",
    root: `${BASE_PATH}artifacts/addb_subject2_walking1_replay`,
    defaultFloorY: 0.008,
    description:
      "AddBiomechanics Wang2023 no-arm Subject 10: walking1_segment_0, extracted from subject2.b3d, with GaitDynamics-predicted GRF/CoP.",
  },
  {
    id: "wang2023-subj10-walking2",
    label: "No-Arms Subject 10, Wang 2023: walking2",
    root: `${BASE_PATH}artifacts/addb_subject2_walking2_replay`,
    defaultFloorY: 0.008,
    description:
      "AddBiomechanics Wang2023 no-arm Subject 10: walking2_segment_0, extracted from subject2.b3d, with GaitDynamics-predicted GRF/CoP.",
  },
  {
    id: "wang2023-subj10-walking3",
    label: "No-Arms Subject 10, Wang 2023: walking3",
    root: `${BASE_PATH}artifacts/addb_subject2_walking3_replay`,
    defaultFloorY: 0.009,
    description:
      "AddBiomechanics Wang2023 no-arm Subject 10: walking3_segment_0, extracted from subject2.b3d, with GaitDynamics-predicted GRF/CoP.",
  },
  {
    id: "wang2023-subj10-walkingTS1",
    label: "No-Arms Subject 10, Wang 2023: walkingTS1",
    root: `${BASE_PATH}artifacts/addb_subject2_walkingTS1_replay`,
    defaultFloorY: 0.012,
    description:
      "AddBiomechanics Wang2023 no-arm Subject 10: walkingTS1_segment_0, extracted from subject2.b3d, with GaitDynamics-predicted GRF/CoP.",
  },
  {
    id: "wang2023-subj10-walkingTS2",
    label: "No-Arms Subject 10, Wang 2023: walkingTS2",
    root: `${BASE_PATH}artifacts/addb_subject2_walkingTS2_replay`,
    defaultFloorY: 0.009,
    description:
      "AddBiomechanics Wang2023 no-arm Subject 10: walkingTS2_segment_0, extracted from subject2.b3d, with GaitDynamics-predicted GRF/CoP.",
  },
  {
    id: "wang2023-subj10-walkingTS4",
    label: "No-Arms Subject 10, Wang 2023: walkingTS4",
    root: `${BASE_PATH}artifacts/addb_subject2_walkingTS4_replay`,
    defaultFloorY: 0.009,
    description:
      "AddBiomechanics Wang2023 no-arm Subject 10: walkingTS4_segment_0, extracted from subject2.b3d, with GaitDynamics-predicted GRF/CoP.",
  },
];
const DEFAULT_MOTION_ID = "opensim-walk";
const selectedMotionArtifact = selectInitialMotionArtifact();
let artifactRoot = selectedMotionArtifact.root;
const FORCE_THRESHOLD_N = 25;
const DEFAULT_PRESSURE_COLOR_MAX_MPA = 0.6;
const DEFAULT_FLOOR_Y = -0.038;
const TERMINAL_STANCE_DROP_FRACTION = 0.45;
const TERMINAL_STANCE_MIN_PROGRESS = 0.58;
const TERMINAL_STANCE_ENVELOPE_MULTIPLIER = 1.4;
const TERMINAL_STANCE_MIN_ENVELOPE_FRACTION = 0.05;
const SOLE_FORCE_DISPLAY_CAP_MULTIPLIER = 1.5;
const REPLAY_ARRAY_KEYS = [
  "bodyTranslations",
  "bodyRotations",
  "rightSoleWorld",
  "leftSoleWorld",
  "rightSoleLocal",
  "leftSoleLocal",
];
const QUAD4_GP = 1 / Math.sqrt(3);
const QUAD4_POINTS = [
  [-QUAD4_GP, -QUAD4_GP],
  [QUAD4_GP, -QUAD4_GP],
  [QUAD4_GP, QUAD4_GP],
  [-QUAD4_GP, QUAD4_GP],
];
const QUAD4_N = QUAD4_POINTS.map(([xi, eta]) => [
  0.25 * (1 - xi) * (1 - eta),
  0.25 * (1 + xi) * (1 - eta),
  0.25 * (1 + xi) * (1 + eta),
  0.25 * (1 - xi) * (1 + eta),
]);
const QUAD4_DN = QUAD4_POINTS.map(([xi, eta]) => [
  [-0.25 * (1 - eta), -0.25 * (1 - xi)],
  [0.25 * (1 - eta), -0.25 * (1 + xi)],
  [0.25 * (1 + eta), 0.25 * (1 + xi)],
  [-0.25 * (1 + eta), 0.25 * (1 - xi)],
]);

const ui = {
  viewport: document.querySelector("#viewport"),
  loading: document.querySelector("#loading"),
  frameReadout: document.querySelector("#frame-readout"),
  playToggle: document.querySelector("#play-toggle"),
  motionSelect: document.querySelector("#motion-select"),
  motionProvenance: document.querySelector("#motion-provenance"),
  scrubber: document.querySelector("#scrubber"),
  floorY: document.querySelector("#floor-y"),
  floorReadout: document.querySelector("#floor-readout"),
  grfChart: document.querySelector("#grf-chart-canvas"),
  legendReference: document.querySelector("#legend-reference"),
  legendGaitDynamics: document.querySelector("#legend-gaitdynamics"),
  floorTuning: document.querySelector("#floor-tuning"),
  soleTop: document.querySelector("#sole-top-canvas"),
  soleLateral: document.querySelector("#sole-lateral-canvas"),
  soleInfoToggle: document.querySelector("#sole-info-toggle"),
  soleInfoPanel: document.querySelector("#sole-info-panel"),
  inferenceMs: document.querySelector("#inference-ms"),
  inferenceHz: document.querySelector("#inference-hz"),
  inferenceUsage: document.querySelector("#inference-usage"),
  activeSoleLabel: document.querySelector("#active-sole-label"),
};

const state = {
  playing: true,
  frame: 0,
  speed: 0.5,
  deformationScale: 1,
  colorMaxMm: DEFAULT_PRESSURE_COLOR_MAX_MPA,
  floorY: DEFAULT_FLOOR_Y,
  geometryOpacity: 0.5,
  activeSide: "right",
  contact: {
    right: null,
    left: null,
  },
  floorAnalysisRequestId: 0,
  floorAnalysisTimer: null,
  floorAnalysisStatus: "",
  lastTimestamp: 0,
  frameAccumulator: 0,
  dirtyFrame: true,
  dirtySoles: true,
  dirtyChart: true,
  nnfe: null,
};

let manifest;
let task1ForceSources;
let arrays;
let derived;
let scene;
let camera;
let renderer;
let controls;
let skeletonLines;
let soleRight;
let soleLeft;
let geometryMeshes = [];
let floorGrid;
let floorMesh;
let rightArrow;
let leftArrow;
let rightCopMarker;
let leftCopMarker;

function selectInitialMotionArtifact() {
  const requested = new URLSearchParams(window.location.search).get("motion");
  const envRoot = import.meta.env.VITE_ARTIFACT_ROOT;
  return (
    MOTION_ARTIFACTS.find((artifact) => artifact.id === requested) ??
    MOTION_ARTIFACTS.find((artifact) => artifact.id === DEFAULT_MOTION_ID) ??
    MOTION_ARTIFACTS.find((artifact) => envRoot && normalizeArtifactRoot(artifact.root) === normalizeArtifactRoot(envRoot)) ??
    MOTION_ARTIFACTS[0]
  );
}

function normalizeArtifactRoot(root) {
  return String(root ?? "").replace(/\/+$/, "");
}

function setupMotionSelector() {
  if (!ui.motionSelect) {
    return;
  }
  ui.motionSelect.replaceChildren(
    ...MOTION_ARTIFACTS.map((artifact) => {
      const option = document.createElement("option");
      option.value = artifact.id;
      option.textContent = artifact.label;
      return option;
    }),
  );
  ui.motionSelect.value = selectedMotionArtifact.id;
  ui.motionSelect.addEventListener("change", () => {
    const next = new URL(window.location.href);
    next.searchParams.set("motion", ui.motionSelect.value);
    next.searchParams.set("v", `${Date.now()}`);
    window.location.href = next.toString();
  });
}

function updateMotionProvenance() {
  if (!ui.motionProvenance) {
    return;
  }
  ui.motionProvenance.textContent = selectedMotionArtifact.description;
}

function initialFloorY() {
  return selectedMotionArtifact.defaultFloorY ?? manifest?.defaults?.floorY ?? DEFAULT_FLOOR_Y;
}

init().catch((error) => {
  const message = error?.message ?? String(error);
  console.error("Unable to load replay", message, error);
  ui.loading.textContent = `Unable to load replay: ${message}`;
});

async function init() {
  setupMotionSelector();
  manifest = await loadManifest();
  task1ForceSources = await loadTask1ForceSources();
  arrays = await loadArrays(manifest.buffers);
  derived = prepareDerivedSeries(manifest, task1ForceSources);
  derived.soleBasisNodes = makeSoleBasisNodeSets(arrays.rightSoleLocal);
  derived.bottomSourceNodeIds = makeBottomSourceNodeMap(arrays.rightSoleLocal, manifest.sole.bottomContactNodeIds ?? []);
  derived.bottomContactIndexByNode = makeBottomContactIndexMap(
    manifest.sole.bottomContactNodeIds ?? [],
    manifest.sole.nodeCount,
  );
  derived.bottomFaceQuadrature = buildBottomFaceQuadrature(
    arrays.rightSoleLocal,
    manifest.sole.bottomContactFaces ?? [],
  );
  state.floorY = initialFloorY();
  state.deformationScale = manifest.defaults?.deformationExaggeration ?? state.deformationScale;
  state.geometryOpacity = manifest.defaults?.geometryOpacity ?? state.geometryOpacity;
  ui.loading.textContent = "Running sole NNFE ONNX inference in browser...";
  await loadNnfeArtifacts();

  applyDefaults();
  updateMotionProvenance();
  setupScene();
  setupUi();
  resize();
  updateFrame();

  ui.loading.classList.add("hidden");
  requestAnimationFrame(tick);
}

async function loadManifest() {
  const response = await fetch(`${artifactRoot}/manifest.json`);
  if (!response.ok) {
    throw new Error(`manifest ${response.status}`);
  }
  return response.json();
}

async function loadTask1ForceSources() {
  const response = await fetch(`${artifactRoot}/task1_force_sources.json`);
  if (!response.ok) {
    return null;
  }
  const data = await response.json();
  if (data.schema !== "gait_nnfe_task1_force_sources.v1") {
    console.warn(`Ignoring unsupported Task 1 force-source schema: ${data.schema}`);
    return null;
  }
  return data;
}

async function loadArrays(buffers) {
  const entries = await Promise.all(
    REPLAY_ARRAY_KEYS.map(async (key) => {
      const spec = buffers[key];
      if (!spec?.path) {
        throw new Error(`missing replay buffer ${key}`);
      }
      const response = await fetch(`${artifactRoot}/${spec.path}`);
      if (!response.ok) {
        throw new Error(`${spec.path} ${response.status}`);
      }
      return [key, new Float32Array(await response.arrayBuffer())];
    }),
  );
  return Object.fromEntries(entries);
}

async function loadNnfeArtifacts() {
  const nnfeSpec = manifest.nnfeModel;
  if (!nnfeSpec?.metadataPath && !nnfeSpec?.modelPath) {
    throw new Error("Missing browser NNFE ONNX artifact");
  }

  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = {
    mjs: ortWasmMjsUrl,
    wasm: ortWasmUrl,
  };

  const metadataFilename = nnfeSpec.metadataPath ?? "sole_floor_contact_metadata.json";
  const metadataResponse = await fetch(`${artifactRoot}/${metadataFilename}`);
  if (!metadataResponse.ok) {
    throw new Error(`${metadataFilename} ${metadataResponse.status}`);
  }
  const metadata = await metadataResponse.json();
  if (metadata.schema !== "gait_sole_floor_contact_onnx.v1") {
    throw new Error(`Unsupported NNFE schema: ${metadata.schema}`);
  }
  const session = await ort.InferenceSession.create(`${artifactRoot}/${metadata.modelPath}`, {
    executionProviders: ["wasm"],
  });
  const started = performance.now();
  const outputs = await runFloorContactReplayBatch(session, metadata);
  const elapsedMs = performance.now() - started;
  const serialStats = await benchmarkSerialFootInference(session, metadata);
  const solveCount = footFrameSolveCount();
  state.nnfe = {
    mode: "floor-contact",
    metadata,
    session,
    outputs,
    elapsedMs,
    solveCount,
    batchInferenceHz: inferenceHzFromElapsed(elapsedMs, solveCount),
    serialEvalMs: serialStats?.meanMs ?? null,
    serialIterations: serialStats?.iterations ?? null,
    inferenceHz: serialStats?.hz ?? inferenceHzFromElapsed(elapsedMs, solveCount),
    live: {
      enabled: true,
      pending: false,
      pendingKey: null,
      queuedFrame: null,
      queuedFloorY: null,
      queuedKey: null,
      lastKey: null,
      lastFrame: null,
      lastFloorY: null,
      lastElapsedMs: null,
    },
  };
  updateInferenceInfo();
  updateDerivedNnfeSeries();
}

async function runFloorContactReplayBatch(session, metadata, floorY = state.floorY) {
  const batch = buildFloorContactFeatureBatch(metadata, floorY);
  const input = new ort.Tensor("float32", batch.features, [manifest.frames * 2, metadata.inputSize]);
  const result = await session.run({ [metadata.inputName]: input });
  const output = result[metadata.outputName] ?? Object.values(result)[0];
  const raw = new Float32Array(output.data);
  return {
    right: extractFloorContactSideOutput(metadata, raw, batch.rightPenetration, 0),
    left: extractFloorContactSideOutput(metadata, raw, batch.leftPenetration, 1),
  };
}

function requestLiveFloorContactInference(frame, force = false) {
  const nnfe = state.nnfe;
  const live = nnfe?.live;
  if (nnfe?.mode !== "floor-contact" || !live?.enabled) {
    return;
  }
  const floorY = state.floorY;
  const key = floorInferenceKey(frame, floorY);
  if (!force && (live.lastKey === key || live.pendingKey === key || live.queuedKey === key)) {
    return;
  }
  live.queuedFrame = frame;
  live.queuedFloorY = floorY;
  live.queuedKey = key;
  if (!live.pending) {
    flushLiveFloorContactInferenceQueue().catch((error) => {
      const message = error?.message ?? String(error);
      console.error("Live floor-contact inference failed", error);
      ui.loading.textContent = `Live floor-contact inference failed: ${message}`;
      ui.loading.classList.remove("hidden");
    });
  }
}

async function flushLiveFloorContactInferenceQueue() {
  const nnfe = state.nnfe;
  const live = nnfe?.live;
  if (!live || live.pending) {
    return;
  }

  while (live.queuedKey && state.nnfe === nnfe) {
    const frame = live.queuedFrame;
    const floorY = live.queuedFloorY;
    const key = live.queuedKey;
    live.queuedFrame = null;
    live.queuedFloorY = null;
    live.queuedKey = null;
    live.pending = true;
    live.pendingKey = key;
    try {
      await runLiveFloorContactInferenceFrame(nnfe, frame, floorY, key);
    } finally {
      live.pending = false;
      live.pendingKey = null;
    }
  }
}

async function runLiveFloorContactInferenceFrame(nnfe, frame, floorY, key) {
  const batch = buildFloorContactFeatureFrameBatch(nnfe.metadata, frame, floorY);
  const input = new ort.Tensor("float32", batch.features, [2, nnfe.metadata.inputSize]);
  const started = performance.now();
  const result = await nnfe.session.run({ [nnfe.metadata.inputName]: input });
  const elapsedMs = performance.now() - started;
  const output = result[nnfe.metadata.outputName] ?? Object.values(result)[0];
  const raw = new Float32Array(output.data);
  if (key !== floorInferenceKey(frame, state.floorY) || state.nnfe !== nnfe) {
    return;
  }

  writeLiveFloorContactSideOutput(nnfe.metadata, raw, batch.rightPenetration, "right", 0, frame);
  writeLiveFloorContactSideOutput(nnfe.metadata, raw, batch.leftPenetration, "left", 1, frame);
  nnfe.live.lastKey = key;
  nnfe.live.lastFrame = frame;
  nnfe.live.lastFloorY = floorY;
  nnfe.live.lastElapsedMs = elapsedMs;
  syncDerivedNnfeFrame(frame);
  updateInferenceInfo();
  state.dirtyFrame = true;
  state.dirtySoles = true;
  state.dirtyChart = true;
  updateFrame();
}

function writeLiveFloorContactSideOutput(metadata, raw, penetrationFrame, side, sideIndex, frame) {
  const output = state.nnfe.outputs[side];
  const nodeCount = manifest.sole.nodeCount;
  const faceCount = derived.bottomFaceQuadrature.length;
  const rawOffset = sideIndex * nodeCount * 3;
  const displacementOffset = frame * nodeCount * 3;
  const pressureOffset = frame * nodeCount;
  const faceOffset = frame * faceCount;
  output.displacementLocal.set(raw.subarray(rawOffset, rawOffset + nodeCount * 3), displacementOffset);

  const fields = computeFloorContactFrameFields(
    metadata,
    output.displacementLocal.subarray(displacementOffset, displacementOffset + nodeCount * 3),
    penetrationFrame,
  );
  output.pressureMpa.set(fields.pressureMpa, pressureOffset);
  if (output.compression !== output.pressureMpa) {
    output.compression.set(fields.pressureMpa, pressureOffset);
  }
  output.facePressureMpa.set(fields.facePressureMpa, faceOffset);
  output.maxCompressionMm[frame] = fields.maxPressureMpa;
  output.maxPressureMpa[frame] = fields.maxPressureMpa;
  output.bottomOutMarginMm[frame] = fields.bottomOutMarginMm;
  output.displacementCompressionMm[frame] = fields.maxCompressionMm;
  output.verticalForceN[frame] = fields.verticalForceN;
  output.activeAreaM2[frame] = fields.activeAreaM2;
}

async function benchmarkSerialFootInference(session, metadata) {
  const batch = buildFloorContactFeatureBatch(metadata);
  const inputSize = metadata.inputSize;
  const gateIndex = metadata.contactGateFeatureIndex ?? inputSize - 2;
  let sourceOffset = 0;
  for (let row = 0; row < batch.features.length / inputSize; row += 1) {
    if (batch.features[row * inputSize + gateIndex] > 0) {
      sourceOffset = row * inputSize;
      break;
    }
  }
  const sample = batch.features.slice(sourceOffset, sourceOffset + inputSize);
  const input = new ort.Tensor("float32", sample, [1, inputSize]);
  await session.run({ [metadata.inputName]: input });
  const iterations = 24;
  const started = performance.now();
  for (let i = 0; i < iterations; i += 1) {
    await session.run({ [metadata.inputName]: input });
  }
  const elapsedMs = performance.now() - started;
  const meanMs = elapsedMs / iterations;
  return {
    iterations,
    elapsedMs,
    meanMs,
    hz: meanMs > 0 ? 1000 / meanMs : 0,
  };
}

function buildFloorContactFeatureBatch(metadata, floorY = state.floorY) {
  const inputSize = metadata.inputSize;
  const bottomIds = metadata.bottomContactNodeIds ?? manifest.sole.bottomContactNodeIds ?? [];
  const bottomCount = bottomIds.length;
  const thicknessM = Math.max(metadata.thicknessM ?? manifest.sole.thicknessM ?? 0.025, 1.0e-9);
  const maxPenetrationM = Math.max(metadata.maxTrainingPenetrationM ?? thicknessM * 2, 1.0e-9);
  const features = new Float32Array(manifest.frames * 2 * inputSize);
  const rightPenetration = new Float32Array(manifest.frames * bottomCount);
  const leftPenetration = new Float32Array(manifest.frames * bottomCount);

  for (let frame = 0; frame < manifest.frames; frame += 1) {
    writeFloorContactFeaturesForSide(
      features,
      rightPenetration,
      frame,
      0,
      inputSize,
      bottomIds,
      thicknessM,
      maxPenetrationM,
      arrays.rightSoleLocal,
      arrays.rightSoleWorld,
      manifest.timeseries.rightForceN[frame]?.[1] ?? 0,
      floorY,
    );
    writeFloorContactFeaturesForSide(
      features,
      leftPenetration,
      frame,
      1,
      inputSize,
      bottomIds,
      thicknessM,
      maxPenetrationM,
      arrays.leftSoleLocal,
      arrays.leftSoleWorld,
      manifest.timeseries.leftForceN[frame]?.[1] ?? 0,
      floorY,
    );
  }

  return { features, rightPenetration, leftPenetration };
}

function buildFloorContactFeatureFrameBatch(metadata, frame, floorY = state.floorY) {
  const inputSize = metadata.inputSize;
  const bottomIds = metadata.bottomContactNodeIds ?? manifest.sole.bottomContactNodeIds ?? [];
  const bottomCount = bottomIds.length;
  const thicknessM = Math.max(metadata.thicknessM ?? manifest.sole.thicknessM ?? 0.025, 1.0e-9);
  const maxPenetrationM = Math.max(metadata.maxTrainingPenetrationM ?? thicknessM * 2, 1.0e-9);
  const features = new Float32Array(2 * inputSize);
  const rightPenetration = new Float32Array(bottomCount);
  const leftPenetration = new Float32Array(bottomCount);

  writeFloorContactFeaturesForSide(
    features,
    rightPenetration,
    frame,
    0,
    inputSize,
    bottomIds,
    thicknessM,
    maxPenetrationM,
    arrays.rightSoleLocal,
    arrays.rightSoleWorld,
    manifest.timeseries.rightForceN[frame]?.[1] ?? 0,
    floorY,
    0,
    0,
  );
  writeFloorContactFeaturesForSide(
    features,
    leftPenetration,
    frame,
    1,
    inputSize,
    bottomIds,
    thicknessM,
    maxPenetrationM,
    arrays.leftSoleLocal,
    arrays.leftSoleWorld,
    manifest.timeseries.leftForceN[frame]?.[1] ?? 0,
    floorY,
    1,
    0,
  );

  return { features, rightPenetration, leftPenetration };
}

function writeFloorContactFeaturesForSide(
  features,
  penetrationTarget,
  frame,
  sideIndex,
  inputSize,
  bottomIds,
  thicknessM,
  maxPenetrationM,
  local,
  world,
  verticalForceN,
  floorY = state.floorY,
  featureRow = frame * 2 + sideIndex,
  penetrationFrame = frame,
) {
  const bottomCount = bottomIds.length;
  const featureOffset = featureRow * inputSize;
  const penetrationOffset = penetrationFrame * bottomCount;
  const nodeCount = manifest.sole.nodeCount;
  const positionOffset = frame * nodeCount * 3;
  const basis = soleBasisFromNodes(local, world, frame);
  const normalLocal = [basis.x.y, basis.y.y, basis.z.y];
  let maxPenetration = 0;
  let totalPenetration = 0;
  const loaded = verticalForceN >= FORCE_THRESHOLD_N;

  for (let i = 0; i < bottomCount; i += 1) {
    const nodeId = bottomIds[i];
    const p = positionOffset + nodeId * 3;
    const signed = world[p + 1] - floorY;
    const rawPenetration = loaded ? Math.max(-signed, 0) : 0;
    const penetration = Math.min(rawPenetration, maxPenetrationM);
    penetrationTarget[penetrationOffset + i] = penetration;
    features[featureOffset + i] = clamp(signed / thicknessM, -2, 2);
    features[featureOffset + bottomCount + i] = clamp(penetration / thicknessM, 0, 2);
    maxPenetration = Math.max(maxPenetration, penetration);
    totalPenetration += penetration;
  }

  features[featureOffset + bottomCount * 2] = normalLocal[0];
  features[featureOffset + bottomCount * 2 + 1] = normalLocal[1];
  features[featureOffset + bottomCount * 2 + 2] = normalLocal[2];
  features[featureOffset + bottomCount * 2 + 3] = maxPenetration / thicknessM;
  features[featureOffset + bottomCount * 2 + 4] = bottomCount ? totalPenetration / bottomCount / thicknessM : 0;
}

function extractFloorContactSideOutput(metadata, raw, penetrationByFrame, sideIndex) {
  const nodeCount = manifest.sole.nodeCount;
  const bottomIds = metadata.bottomContactNodeIds ?? manifest.sole.bottomContactNodeIds ?? [];
  const bottomCount = bottomIds.length;
  const displacementLocal = new Float32Array(manifest.frames * nodeCount * 3);
  const pressureMpa = new Float32Array(manifest.frames * nodeCount);
  const facePressureMpa = new Float32Array(manifest.frames * derived.bottomFaceQuadrature.length);
  const maxPressureMpa = new Float32Array(manifest.frames);
  const bottomOutMarginMm = new Float32Array(manifest.frames);
  const maxCompressionMm = new Float32Array(manifest.frames);
  const verticalForceN = new Float32Array(manifest.frames);
  const activeAreaM2 = new Float32Array(manifest.frames);

  for (let frame = 0; frame < manifest.frames; frame += 1) {
    const batchIndex = frame * 2 + sideIndex;
    const rawOffset = batchIndex * nodeCount * 3;
    const dstOffset = frame * nodeCount * 3;

    for (let p = 0; p < nodeCount * 3; p += 1) {
      displacementLocal[dstOffset + p] = raw[rawOffset + p];
    }

    const fields = computeFloorContactFrameFields(
      metadata,
      displacementLocal.subarray(dstOffset, dstOffset + nodeCount * 3),
      penetrationByFrame.subarray(frame * bottomCount, (frame + 1) * bottomCount),
    );
    const pressureOffset = frame * nodeCount;
    pressureMpa.set(fields.pressureMpa, pressureOffset);
    facePressureMpa.set(fields.facePressureMpa, frame * derived.bottomFaceQuadrature.length);
    maxPressureMpa[frame] = fields.maxPressureMpa;
    maxCompressionMm[frame] = fields.maxCompressionMm;
    bottomOutMarginMm[frame] = fields.bottomOutMarginMm;
    verticalForceN[frame] = fields.verticalForceN;
    activeAreaM2[frame] = fields.activeAreaM2;
  }

  return {
    displacementLocal,
    compression: pressureMpa,
    pressureMpa,
    facePressureMpa,
    maxCompressionMm: maxPressureMpa,
    maxPressureMpa,
    bottomOutMarginMm,
    displacementCompressionMm: maxCompressionMm,
    verticalForceN,
    activeAreaM2,
  };
}

function computeFloorContactFrameFields(metadata, displacementFrame, penetrationFrame) {
  const nodeCount = manifest.sole.nodeCount;
  const bottomIds = metadata.bottomContactNodeIds ?? manifest.sole.bottomContactNodeIds ?? [];
  const pressureMpa = new Float32Array(nodeCount);
  const facePressureMpa = new Float32Array(derived.bottomFaceQuadrature.length);
  const nodePressureWeighted = new Float32Array(nodeCount);
  const nodePressureWeights = new Float32Array(nodeCount);
  const bottomPressure = new Float32Array(nodeCount);
  const thicknessMm = (metadata.thicknessM ?? manifest.sole.thicknessM ?? 0.025) * 1000;
  const contactStiffness = metadata.contactStiffnessMpaPerMm ?? 0.05;
  let maxPressureMpa = 0;
  let maxCompressionMm = 0;
  let verticalForceN = 0;
  let activeAreaM2 = 0;

  for (const nodeId of bottomIds) {
    maxCompressionMm = Math.max(maxCompressionMm, Math.max(displacementFrame[nodeId * 3 + 1], 0) * 1000);
  }

  for (let faceIndex = 0; faceIndex < derived.bottomFaceQuadrature.length; faceIndex += 1) {
    const face = derived.bottomFaceQuadrature[faceIndex];
    let facePressureWeighted = 0;
    let faceAreaWeight = 0;

    for (let gp = 0; gp < QUAD4_N.length; gp += 1) {
      const shape = QUAD4_N[gp];
      const jacWeight = face.jacobian[gp];
      let penetrationGpMm = 0;
      let upwardGpMm = 0;

      for (let localNode = 0; localNode < 4; localNode += 1) {
        const nodeId = face.nodeIds[localNode];
        const bottomIndex = derived.bottomContactIndexByNode[nodeId];
        const n = shape[localNode];
        if (bottomIndex >= 0) {
          penetrationGpMm += n * penetrationFrame[bottomIndex] * 1000;
        }
        upwardGpMm += n * Math.max(displacementFrame[nodeId * 3 + 1], 0) * 1000;
      }

      const residualMm = Math.max(penetrationGpMm - upwardGpMm, 0);
      const pressure = residualMm * contactStiffness;
      facePressureWeighted += pressure * jacWeight;
      faceAreaWeight += jacWeight;
      maxPressureMpa = Math.max(maxPressureMpa, pressure);
      verticalForceN += pressure * 1.0e6 * jacWeight;
      if (residualMm > 1.0e-6) {
        activeAreaM2 += jacWeight;
      }

      for (let localNode = 0; localNode < 4; localNode += 1) {
        const nodeId = face.nodeIds[localNode];
        const weight = shape[localNode] * jacWeight;
        nodePressureWeighted[nodeId] += pressure * weight;
        nodePressureWeights[nodeId] += weight;
      }
    }

    facePressureMpa[faceIndex] = faceAreaWeight > 0 ? facePressureWeighted / faceAreaWeight : 0;
  }

  for (const nodeId of bottomIds) {
    bottomPressure[nodeId] = nodePressureWeights[nodeId] > 0
      ? nodePressureWeighted[nodeId] / nodePressureWeights[nodeId]
      : 0;
  }

  for (let node = 0; node < nodeCount; node += 1) {
    const sourceNode = derived.bottomSourceNodeIds[node] ?? node;
    pressureMpa[node] = bottomPressure[sourceNode] ?? 0;
  }

  return {
    pressureMpa,
    facePressureMpa,
    maxPressureMpa,
    maxCompressionMm,
    bottomOutMarginMm: thicknessMm - maxCompressionMm,
    verticalForceN,
    activeAreaM2,
  };
}

function updateDerivedNnfeSeries() {
  const outputs = state.nnfe?.outputs;
  if (!outputs) {
    return;
  }
  derived.rightMaxCompressionMm = outputs.right.maxCompressionMm;
  derived.leftMaxCompressionMm = outputs.left.maxCompressionMm;
  derived.rightBottomOutMarginMm = outputs.right.bottomOutMarginMm;
  derived.leftBottomOutMarginMm = outputs.left.bottomOutMarginMm;
  derived.rightSoleForceY = outputs.right.verticalForceN;
  derived.leftSoleForceY = outputs.left.verticalForceN;
  derived.rightSoleActiveAreaM2 = outputs.right.activeAreaM2;
  derived.leftSoleActiveAreaM2 = outputs.left.activeAreaM2;
  updateSoleForceDisplaySeries();
  derived.maxCompression = Math.max(1, ...outputs.right.maxCompressionMm, ...outputs.left.maxCompressionMm);
  derived.minMargin = Math.min(...outputs.right.bottomOutMarginMm, ...outputs.left.bottomOutMarginMm);
  derived.maxMargin = Math.max(...outputs.right.bottomOutMarginMm, ...outputs.left.bottomOutMarginMm);
  derived.maxForce = Math.max(
    1,
    ...(derived.forceScaleValues ?? []),
    ...(derived.rightSoleForceDisplayY ?? []),
    ...(derived.leftSoleForceDisplayY ?? []),
  );
  updateFloorTuningReadout();
}

function syncDerivedNnfeFrame(frame) {
  const outputs = state.nnfe?.outputs;
  if (!outputs) {
    return;
  }
  derived.rightMaxCompressionMm[frame] = outputs.right.maxCompressionMm[frame];
  derived.leftMaxCompressionMm[frame] = outputs.left.maxCompressionMm[frame];
  derived.rightBottomOutMarginMm[frame] = outputs.right.bottomOutMarginMm[frame];
  derived.leftBottomOutMarginMm[frame] = outputs.left.bottomOutMarginMm[frame];
  derived.rightSoleForceY[frame] = outputs.right.verticalForceN[frame];
  derived.leftSoleForceY[frame] = outputs.left.verticalForceN[frame];
  derived.rightSoleActiveAreaM2[frame] = outputs.right.activeAreaM2[frame];
  derived.leftSoleActiveAreaM2[frame] = outputs.left.activeAreaM2[frame];
  updateSoleForceDisplaySeries();
  derived.maxCompression = Math.max(
    derived.maxCompression ?? 1,
    outputs.right.maxCompressionMm[frame] ?? 0,
    outputs.left.maxCompressionMm[frame] ?? 0,
  );
  derived.minMargin = Math.min(
    derived.minMargin ?? Infinity,
    outputs.right.bottomOutMarginMm[frame] ?? Infinity,
    outputs.left.bottomOutMarginMm[frame] ?? Infinity,
  );
  derived.maxMargin = Math.max(
    derived.maxMargin ?? -Infinity,
    outputs.right.bottomOutMarginMm[frame] ?? -Infinity,
    outputs.left.bottomOutMarginMm[frame] ?? -Infinity,
  );
  derived.maxForce = Math.max(
    1,
    ...(derived.forceScaleValues ?? []),
    ...(derived.rightSoleForceDisplayY ?? []),
    ...(derived.leftSoleForceDisplayY ?? []),
  );
  updateFloorTuningReadout();
}

function updateSoleForceDisplaySeries() {
  if (!derived?.rightSoleForceY || !derived?.leftSoleForceY) {
    return;
  }
  const target = getSoleForceReferenceTarget();
  const displayCapN = getSoleForceDisplayCap(target);
  const right = filterTerminalStanceForceForDisplay(derived.rightSoleForceY, target.rightForceY, displayCapN);
  const left = filterTerminalStanceForceForDisplay(derived.leftSoleForceY, target.leftForceY, displayCapN);
  derived.rightSoleForceDisplayY = right.data;
  derived.leftSoleForceDisplayY = left.data;
  derived.soleForceClipInfo = {
    displayCapN,
    rightFilteredFrames: right.filteredFrames,
    leftFilteredFrames: left.filteredFrames,
    rightCappedFrames: right.cappedFrames,
    leftCappedFrames: left.cappedFrames,
    rightRawPeakN: right.rawPeakN,
    leftRawPeakN: left.rawPeakN,
    rightDisplayPeakN: right.displayPeakN,
    leftDisplayPeakN: left.displayPeakN,
  };
}

function getSoleForceDisplayCap(target) {
  const capSource = derived?.task1?.predicted ?? target;
  const rightPeak = maxSeriesValue(capSource.rightForceY);
  const leftPeak = maxSeriesValue(capSource.leftForceY);
  const targetPeak = Math.max(
    rightPeak,
    leftPeak,
    1,
  );
  return targetPeak * SOLE_FORCE_DISPLAY_CAP_MULTIPLIER;
}

function filterTerminalStanceForceForDisplay(source, reference, displayCapN = Infinity) {
  const data = Float32Array.from(source);
  const rawPeakN = maxSeriesValue(source);
  const referencePeakN = maxSeriesValue(reference);
  let filteredFrames = 0;

  if (!reference?.length || referencePeakN <= FORCE_THRESHOLD_N) {
    return {
      data,
      filteredFrames,
      rawPeakN,
      displayPeakN: rawPeakN,
      cappedFrames: 0,
    };
  }

  const stanceThresholdN = Math.max(FORCE_THRESHOLD_N, referencePeakN * 0.05);
  const intervals = findStanceIntervals(reference, stanceThresholdN);
  for (const interval of intervals) {
    const stanceFrames = interval.end - interval.start + 1;
    if (stanceFrames < 5) {
      continue;
    }

    const peakIndex = findPeakIndex(reference, interval.start, interval.end);
    const minTailStart = interval.start + Math.floor(stanceFrames * TERMINAL_STANCE_MIN_PROGRESS);
    let tailStart = interval.end + 1;
    for (let i = Math.max(peakIndex + 1, minTailStart); i <= interval.end; i += 1) {
      if ((reference[i] ?? 0) <= reference[peakIndex] * TERMINAL_STANCE_DROP_FRACTION) {
        tailStart = i;
        break;
      }
    }
    if (tailStart > interval.end) {
      tailStart = interval.start + Math.floor(stanceFrames * 0.75);
    }

    const tailEnd = Math.min(source.length - 1, interval.end + 2);
    for (let i = tailStart; i <= tailEnd; i += 1) {
      const refValue = reference[i] ?? 0;
      const envelopeN = Math.max(
        refValue * TERMINAL_STANCE_ENVELOPE_MULTIPLIER,
        referencePeakN * TERMINAL_STANCE_MIN_ENVELOPE_FRACTION,
      );
      if (data[i] > envelopeN) {
        data[i] = envelopeN;
        filteredFrames += 1;
      }
    }
  }

  const cappedFrames = capForceSeriesForDisplay(data, displayCapN);

  return {
    data,
    filteredFrames,
    cappedFrames,
    rawPeakN,
    displayPeakN: maxSeriesValue(data),
  };
}

function capForceSeriesForDisplay(data, displayCapN) {
  if (!Number.isFinite(displayCapN)) {
    return 0;
  }
  let cappedFrames = 0;
  for (let i = 0; i < data.length; i += 1) {
    if (data[i] > displayCapN) {
      data[i] = displayCapN;
      cappedFrames += 1;
    }
  }
  return cappedFrames;
}

function findStanceIntervals(forceY, thresholdN) {
  const intervals = [];
  let start = null;
  for (let i = 0; i < forceY.length; i += 1) {
    const active = (forceY[i] ?? 0) >= thresholdN;
    if (active && start === null) {
      start = i;
    } else if (!active && start !== null) {
      intervals.push({ start, end: i - 1 });
      start = null;
    }
  }
  if (start !== null) {
    intervals.push({ start, end: forceY.length - 1 });
  }
  return mergeNearbyIntervals(intervals, 2);
}

function mergeNearbyIntervals(intervals, maxGapFrames) {
  const merged = [];
  for (const interval of intervals) {
    const previous = merged[merged.length - 1];
    if (previous && interval.start - previous.end <= maxGapFrames + 1) {
      previous.end = interval.end;
    } else {
      merged.push({ ...interval });
    }
  }
  return merged;
}

function findPeakIndex(series, start, end) {
  let peakIndex = start;
  let peakValue = Number(series[start]) || 0;
  for (let i = start + 1; i <= end; i += 1) {
    const value = Number(series[i]) || 0;
    if (value > peakValue) {
      peakIndex = i;
      peakValue = value;
    }
  }
  return peakIndex;
}

function maxSeriesValue(series) {
  if (!series?.length) {
    return 0;
  }
  let maxValue = 0;
  for (let i = 0; i < series.length; i += 1) {
    maxValue = Math.max(maxValue, Number(series[i]) || 0);
  }
  return maxValue;
}

function applyDefaults() {
  state.deformationScale = manifest.defaults?.deformationExaggeration ?? state.deformationScale;
  state.colorMaxMm = DEFAULT_PRESSURE_COLOR_MAX_MPA;
  state.floorY = initialFloorY();
  state.geometryOpacity = manifest.defaults?.geometryOpacity ?? state.geometryOpacity;

  ui.scrubber.max = `${manifest.frames - 1}`;
  ui.floorY.value = `${state.floorY}`;
  updateControlReadouts();
}

function prepareDerivedSeries(data, task1Data = null) {
  const rightForceY = data.timeseries.rightForceN.map((force) => force[1] ?? 0);
  const leftForceY = data.timeseries.leftForceN.map((force) => force[1] ?? 0);
  const rightForceNorm = data.timeseries.rightForceN.map(vectorNorm);
  const leftForceNorm = data.timeseries.leftForceN.map(vectorNorm);
  const task1 = prepareTask1ForceSeries(task1Data);
  const task1ForceValues = task1 ? forceValuesForTask1(task1) : [];
  const maxForce = Math.max(1, ...rightForceY, ...leftForceY, ...task1ForceValues);
  const forceScaleValues = [
    ...rightForceY,
    ...leftForceY,
    ...task1ForceValues,
  ];
  const maxCompression = Math.max(
    1,
    ...data.timeseries.rightMaxCompressionMm,
    ...data.timeseries.leftMaxCompressionMm,
  );
  const marginValues = [
    ...data.timeseries.rightBottomOutMarginMm,
    ...data.timeseries.leftBottomOutMarginMm,
  ];
  return {
    fps: data.durationS > 0 ? (data.frames - 1) / data.durationS : 60,
    rightForceY,
    leftForceY,
    rightForceNorm,
    leftForceNorm,
    forceScaleValues,
    rightSoleForceY: new Float32Array(data.frames),
    leftSoleForceY: new Float32Array(data.frames),
    rightSoleForceDisplayY: new Float32Array(data.frames),
    leftSoleForceDisplayY: new Float32Array(data.frames),
    soleForceClipInfo: null,
    rightSoleActiveAreaM2: new Float32Array(data.frames),
    leftSoleActiveAreaM2: new Float32Array(data.frames),
    rightMaxCompressionMm: Float32Array.from(data.timeseries.rightMaxCompressionMm),
    leftMaxCompressionMm: Float32Array.from(data.timeseries.leftMaxCompressionMm),
    rightBottomOutMarginMm: Float32Array.from(data.timeseries.rightBottomOutMarginMm),
    leftBottomOutMarginMm: Float32Array.from(data.timeseries.leftBottomOutMarginMm),
    maxForce,
    maxCompression,
    minMargin: Math.min(...marginValues),
    maxMargin: Math.max(...marginValues),
    quadEdges: buildFaceEdges(data.sole.quadFaces ?? data.sole.faces),
    task1,
  };
}

function prepareTask1ForceSeries(task1Data) {
  const sources = task1Data?.sources;
  const predicted = sources?.gaitDynamicsPredicted ? prepareForceSource(sources.gaitDynamicsPredicted) : null;
  if (!predicted) {
    return null;
  }
  return {
    provenance: task1Data.provenance ?? {},
    metrics: task1Data.metrics ?? {},
    reference: null,
    predicted,
  };
}

function forceValuesForTask1(task1) {
  const values = [];
  for (const source of [task1.predicted]) {
    if (!source) {
      continue;
    }
    values.push(...source.rightForceY, ...source.leftForceY);
  }
  return values;
}

function prepareForceSource(source) {
  return {
    label: source.label,
    rightForceN: source.rightForceN ?? [],
    leftForceN: source.leftForceN ?? [],
    rightCopM: source.rightCopM ?? [],
    leftCopM: source.leftCopM ?? [],
    rightForceY: (source.rightForceN ?? []).map((force) => force[1] ?? 0),
    leftForceY: (source.leftForceN ?? []).map((force) => force[1] ?? 0),
    rightForceNorm: (source.rightForceN ?? []).map(vectorNorm),
    leftForceNorm: (source.leftForceN ?? []).map(vectorNorm),
  };
}

function setupScene() {
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf4f4f2);

  renderer = new THREE.WebGLRenderer({
    canvas: ui.viewport,
    antialias: true,
    alpha: false,
    powerPreference: "high-performance",
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  const bounds = makeBounds(manifest.bounds);
  camera = new THREE.PerspectiveCamera(46, 1, 0.01, 100);
  const viewDistance = Math.max(3.8, bounds.span.x * 1.25);
  camera.position.set(bounds.center.x + 0.18, bounds.center.y + 0.92, bounds.center.z + viewDistance);

  controls = new OrbitControls(camera, ui.viewport);
  controls.target.copy(bounds.center);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;
  controls.maxPolarAngle = Math.PI * 0.92;

  scene.add(new THREE.HemisphereLight(0xffffff, 0xc9c9c9, 1.35));
  const key = new THREE.DirectionalLight(0xffffff, 1.25);
  key.position.set(bounds.center.x - 2.5, bounds.center.y + 4, bounds.center.z + 3);
  scene.add(key);

  const fill = new THREE.DirectionalLight(0xffffff, 0.45);
  fill.position.set(bounds.center.x + 3, bounds.center.y + 1.5, bounds.center.z - 2);
  scene.add(fill);

  const gridSize = Math.max(3.2, bounds.span.x + 0.6, bounds.span.z + 0.6);
  floorGrid = new THREE.GridHelper(gridSize, 26, 0x858585, 0xd0d0d0);
  floorGrid.position.set(bounds.center.x, state.floorY, bounds.center.z);
  scene.add(floorGrid);

  floorMesh = new THREE.Mesh(
    new THREE.PlaneGeometry(gridSize, gridSize),
    new THREE.MeshBasicMaterial({
      color: 0xf8f8f6,
      transparent: true,
      opacity: 0.82,
      side: THREE.DoubleSide,
      depthWrite: false,
    }),
  );
  floorMesh.rotation.x = -Math.PI / 2;
  floorMesh.position.set(bounds.center.x, state.floorY, bounds.center.z);
  scene.add(floorMesh);

  skeletonLines = createSkeletonLines();
  scene.add(skeletonLines);

  soleRight = createSoleMesh(0x2f6f9f);
  soleLeft = createSoleMesh(0xb26a2a);
  scene.add(soleRight.group, soleLeft.group);

  geometryMeshes = createGeometryMeshes();
  geometryMeshes.forEach((mesh) => scene.add(mesh));

  rightArrow = createArrow(0x2f6f9f);
  leftArrow = createArrow(0xb26a2a);
  scene.add(rightArrow, leftArrow);

  rightCopMarker = createCopMarker(0x2f6f9f);
  leftCopMarker = createCopMarker(0xb26a2a);
  scene.add(rightCopMarker, leftCopMarker);

  window.addEventListener("resize", resize);
}

function makeBounds(bounds) {
  const min = new THREE.Vector3(...bounds.min);
  const max = new THREE.Vector3(...bounds.max);
  const center = new THREE.Vector3().addVectors(min, max).multiplyScalar(0.5);
  const span = new THREE.Vector3().subVectors(max, min);
  center.y = Math.max(0.32, center.y);
  return { min, max, center, span };
}

function setupUi() {
  updateLegendVisibility();

  ui.playToggle.addEventListener("click", () => {
    state.playing = !state.playing;
    ui.playToggle.textContent = state.playing ? "Pause" : "Play";
  });

  if (ui.soleInfoToggle && ui.soleInfoPanel) {
    ui.soleInfoToggle.addEventListener("click", () => {
      const expanded = ui.soleInfoToggle.getAttribute("aria-expanded") === "true";
      ui.soleInfoToggle.setAttribute("aria-expanded", `${!expanded}`);
      ui.soleInfoPanel.hidden = expanded;
    });
  }

  ui.scrubber.addEventListener("input", () => {
    state.frame = Number(ui.scrubber.value);
    state.frameAccumulator = 0;
    state.dirtyFrame = true;
    state.dirtyChart = true;
  });

  ui.floorY.addEventListener("input", () => {
    state.floorY = Number(ui.floorY.value);
    updateFloorPlane();
    state.dirtyFrame = true;
    state.dirtySoles = true;
    state.dirtyChart = true;
    updateControlReadouts();
    scheduleFloorAnalysisRecompute();
    requestLiveFloorContactInference(state.frame, true);
  });

  updateInferenceInfo();
  updateFloorTuningReadout();
}

function createSkeletonLines() {
  const edgeCount = manifest.skeletonEdges.length;
  const positions = new Float32Array(edgeCount * 2 * 3);
  const colors = new Float32Array(edgeCount * 2 * 3);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3).setUsage(THREE.DynamicDrawUsage));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

  manifest.skeletonEdges.forEach((edge, edgeIndex) => {
    const color = parseEdgeColor(edge.color);
    for (let endpoint = 0; endpoint < 2; endpoint += 1) {
      const offset = (edgeIndex * 2 + endpoint) * 3;
      colors[offset] = color.r;
      colors[offset + 1] = color.g;
      colors[offset + 2] = color.b;
    }
  });

  const material = new THREE.LineBasicMaterial({
    vertexColors: true,
    linewidth: 2,
    transparent: true,
    opacity: 0.95,
  });
  return new THREE.LineSegments(geometry, material);
}

function createSoleMesh(edgeColor) {
  const nodeCount = manifest.sole.nodeCount;
  const positions = new Float32Array(nodeCount * 3);
  const colors = new Float32Array(nodeCount * 3);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3).setUsage(THREE.DynamicDrawUsage));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3).setUsage(THREE.DynamicDrawUsage));
  geometry.setIndex(flattenFaces(manifest.sole.triangleFaces ?? triangulateQuads(manifest.sole.faces), nodeCount));
  geometry.computeVertexNormals();

  const material = new THREE.MeshStandardMaterial({
    vertexColors: true,
    roughness: 0.78,
    metalness: 0.0,
    side: THREE.DoubleSide,
    transparent: true,
    opacity: 0.96,
  });

  const mesh = new THREE.Mesh(geometry, material);
  const wirePositions = new Float32Array(derived.quadEdges.length * 2 * 3);
  const wireGeometry = new THREE.BufferGeometry();
  wireGeometry.setAttribute("position", new THREE.BufferAttribute(wirePositions, 3).setUsage(THREE.DynamicDrawUsage));
  const wire = new THREE.LineSegments(
    wireGeometry,
    new THREE.LineBasicMaterial({
      color: edgeColor,
      transparent: true,
      opacity: 0.5,
      depthWrite: false,
    }),
  );

  const group = new THREE.Group();
  group.add(mesh, wire);
  return {
    group,
    mesh,
    wire,
    geometry,
    wireGeometry,
    positions,
    colors,
    wirePositions,
  };
}

function flattenFaces(faces, nodeCount) {
  const IndexArray = nodeCount > 65535 ? Uint32Array : Uint16Array;
  const index = new IndexArray(faces.length * 3);
  faces.forEach((face, faceIndex) => {
    const offset = faceIndex * 3;
    index[offset] = face[0];
    index[offset + 1] = face[1];
    index[offset + 2] = face[2];
  });
  return new THREE.BufferAttribute(index, 1);
}

function triangulateQuads(faces) {
  const triangles = [];
  for (const face of faces) {
    if (face.length === 3) {
      triangles.push(face);
    } else if (face.length >= 4) {
      triangles.push([face[0], face[1], face[2]]);
      triangles.push([face[0], face[2], face[3]]);
    }
  }
  return triangles;
}

function createGeometryMeshes() {
  const materialCache = new Map();
  return manifest.geometryMeshes.map((meshSpec) => {
    const vertices = Float32Array.from(meshSpec.vertices.flat());
    const vertexCount = vertices.length / 3;
    const triangles = meshSpec.triangles.flat();
    const IndexArray = vertexCount > 65535 ? Uint32Array : Uint16Array;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(vertices, 3));
    geometry.setIndex(new THREE.BufferAttribute(IndexArray.from(triangles), 1));
    geometry.computeVertexNormals();

    const bodyName = meshSpec.body;
    const color = bodyName.includes("_r")
      ? 0xc8d5df
      : bodyName.includes("_l")
        ? 0xd8d0c7
        : 0xd4d4d0;
    const materialKey = `${color}`;
    if (!materialCache.has(materialKey)) {
      materialCache.set(
        materialKey,
        new THREE.MeshStandardMaterial({
          color,
          transparent: true,
          opacity: state.geometryOpacity,
          roughness: 0.92,
          metalness: 0.0,
          side: THREE.DoubleSide,
          depthWrite: true,
        }),
      );
    }
    const mesh = new THREE.Mesh(geometry, materialCache.get(materialKey));
    mesh.matrixAutoUpdate = false;
    mesh.userData.bodyIndex = meshSpec.bodyIndex;
    return mesh;
  });
}

function createArrow(color) {
  const arrow = new THREE.ArrowHelper(
    new THREE.Vector3(0, 1, 0),
    new THREE.Vector3(0, 0, 0),
    0.1,
    color,
    0.08,
    0.04,
  );
  arrow.visible = false;
  return arrow;
}

function createCopMarker(color) {
  const marker = new THREE.Mesh(
    new THREE.SphereGeometry(0.018, 24, 12),
    new THREE.MeshStandardMaterial({
      color,
      emissive: 0x000000,
      emissiveIntensity: 0,
      roughness: 0.55,
    }),
  );
  marker.visible = false;
  return marker;
}

function tick(timestamp) {
  if (!state.lastTimestamp) {
    state.lastTimestamp = timestamp;
  }
  const dt = Math.min(0.08, (timestamp - state.lastTimestamp) / 1000);
  state.lastTimestamp = timestamp;

  if (state.playing) {
    state.frameAccumulator += dt * derived.fps * state.speed;
    const framesToAdvance = Math.floor(state.frameAccumulator);
    if (framesToAdvance > 0) {
      state.frame = (state.frame + framesToAdvance) % manifest.frames;
      state.frameAccumulator -= framesToAdvance;
      state.dirtyFrame = true;
      state.dirtyChart = true;
    }
  }

  updateFrame();
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}

function updateFrame() {
  if (!state.dirtyFrame && !state.dirtySoles && !state.dirtyChart) {
    return;
  }

  const frame = state.frame;
  requestLiveFloorContactInference(frame);
  ui.scrubber.value = `${frame}`;
  ui.frameReadout.textContent = `t=${formatNumber(manifest.timeseries.time[frame], 2)} s · frame ${frame + 1}/${manifest.frames}`;

  updateSkeleton(frame);
  updateBodyGeometry(frame);
  updateSoles(frame);
  updateForceMarkers(frame);
  updateMetrics(frame);
  if (state.dirtyChart) {
    drawInstrumentation(frame);
  }

  state.dirtyFrame = false;
  state.dirtySoles = false;
  state.dirtyChart = false;
}

function updateSkeleton(frame) {
  const positions = skeletonLines.geometry.attributes.position.array;
  manifest.skeletonEdges.forEach((edge, edgeIndex) => {
    const startIndex = manifest.bodyNames.indexOf(edge.start);
    const endIndex = manifest.bodyNames.indexOf(edge.end);
    writeBodyTranslation(positions, edgeIndex * 6, frame, startIndex);
    writeBodyTranslation(positions, edgeIndex * 6 + 3, frame, endIndex);
  });
  skeletonLines.geometry.attributes.position.needsUpdate = true;
}

function updateBodyGeometry(frame) {
  const matrix = new THREE.Matrix4();
  geometryMeshes.forEach((mesh) => {
    const bodyIndex = mesh.userData.bodyIndex;
    setMatrixFromBody(frame, bodyIndex, matrix);
    mesh.matrix.copy(matrix);
  });
}

function updateSoles(frame) {
  if (!state.dirtyFrame && !state.dirtySoles) {
    return;
  }
  const outputs = state.nnfe?.outputs;
  if (!outputs) {
    return;
  }
  state.contact.right = updateSoleGeometry(
    soleRight,
    "right",
    arrays.rightSoleWorld,
    arrays.rightSoleLocal,
    outputs.right.displacementLocal,
    outputs.right.compression,
    frame,
  );
  state.contact.left = updateSoleGeometry(
    soleLeft,
    "left",
    arrays.leftSoleWorld,
    arrays.leftSoleLocal,
    outputs.left.displacementLocal,
    outputs.left.compression,
    frame,
  );
}

function updateSoleGeometry(sole, side, world, local, displacementLocal, compression, frame) {
  const nodeCount = manifest.sole.nodeCount;
  const positionOffset = frame * nodeCount * 3;
  const compressionOffset = frame * nodeCount;
  const basis = soleBasisFromNodes(local, world, frame);
  for (let node = 0; node < nodeCount; node += 1) {
    const p = node * 3;
    const src = positionOffset + p;
    const dx = displacementLocal[src] * state.deformationScale;
    const dy = displacementLocal[src + 1] * state.deformationScale;
    const dz = displacementLocal[src + 2] * state.deformationScale;
    sole.positions[p] = world[src] + basis.x.x * dx + basis.y.x * dy + basis.z.x * dz;
    sole.positions[p + 1] = world[src + 1] + basis.x.y * dx + basis.y.y * dy + basis.z.y * dz;
    sole.positions[p + 2] = world[src + 2] + basis.x.z * dx + basis.y.z * dy + basis.z.z * dz;
    writeCompressionColor(sole.colors, p, compression[compressionOffset + node]);
  }
  updateQuadWireGeometry(sole);
  const contactSummary = updateContactGeometry(sole, compression, frame);
  sole.geometry.attributes.position.needsUpdate = true;
  sole.geometry.attributes.color.needsUpdate = true;
  sole.geometry.computeVertexNormals();
  sole.geometry.computeBoundingSphere();
  return contactSummary;
}

function updateQuadWireGeometry(sole) {
  let target = 0;
  for (const [aIndex, bIndex] of derived.quadEdges) {
    const a = aIndex * 3;
    const b = bIndex * 3;
    sole.wirePositions[target] = sole.positions[a];
    sole.wirePositions[target + 1] = sole.positions[a + 1];
    sole.wirePositions[target + 2] = sole.positions[a + 2];
    sole.wirePositions[target + 3] = sole.positions[b];
    sole.wirePositions[target + 4] = sole.positions[b + 1];
    sole.wirePositions[target + 5] = sole.positions[b + 2];
    target += 6;
  }
  sole.wireGeometry.attributes.position.needsUpdate = true;
}

function updateContactGeometry(sole, colorField = null, frame = 0) {
  const ids = manifest.sole.bottomContactNodeIds ?? [];
  const tolerance = manifest.sole.contactToleranceM ?? 0.002;
  const nodeCount = manifest.sole.nodeCount;
  const colorOffset = frame * nodeCount;
  const contactByNode = new Map();
  let contactCount = 0;
  let minClearanceM = Infinity;
  let maxPenetrationM = 0;

  for (const nodeId of ids) {
    const p = nodeId * 3;
    const clearanceM = sole.positions[p + 1] - state.floorY;
    const penetrationM = Math.max(-clearanceM, 0);
    const colorValue = colorField ? colorField[colorOffset + nodeId] : 0;
    minClearanceM = Math.min(minClearanceM, clearanceM);
    maxPenetrationM = Math.max(maxPenetrationM, penetrationM);
    const touching = colorValue > 1.0e-4 || clearanceM <= tolerance;
    contactByNode.set(nodeId, { clearanceM, penetrationM, touching });
    if (!touching) {
      continue;
    }
    contactCount += 1;
  }

  return {
    contactByNode,
    contactCount,
    minClearanceM: Number.isFinite(minClearanceM) ? minClearanceM : 0,
    maxPenetrationM,
    toleranceM: tolerance,
  };
}

function updateForceMarkers(frame) {
  const markerSource = derived.task1?.predicted;
  const rightForce = markerSource?.rightForceN?.[frame] ?? manifest.timeseries.rightForceN[frame];
  const rightCop = markerSource?.rightCopM?.[frame] ?? manifest.timeseries.rightProjectedCopM[frame];
  const rightNorm = markerSource?.rightForceNorm?.[frame] ?? derived.rightForceNorm[frame];
  const leftForce = markerSource?.leftForceN?.[frame] ?? manifest.timeseries.leftForceN[frame];
  const leftCop = markerSource?.leftCopM?.[frame] ?? manifest.timeseries.leftProjectedCopM[frame];
  const leftNorm = markerSource?.leftForceNorm?.[frame] ?? derived.leftForceNorm[frame];
  updateArrowAndCop(
    rightArrow,
    rightCopMarker,
    rightForce,
    rightCop,
    rightNorm,
  );
  updateArrowAndCop(
    leftArrow,
    leftCopMarker,
    leftForce,
    leftCop,
    leftNorm,
  );
}

function updateArrowAndCop(arrow, marker, force, cop, forceNorm) {
  const visibleForce = forceNorm > FORCE_THRESHOLD_N;
  arrow.visible = visibleForce;
  if (visibleForce) {
    const origin = new THREE.Vector3(...cop);
    const direction = new THREE.Vector3(...force).normalize();
    arrow.position.copy(origin);
    arrow.setDirection(direction);
    arrow.setLength(Math.min(0.62, Math.max(0.08, forceNorm * 0.0005)), 0.08, 0.045);
  }

  const visibleCop = forceNorm > FORCE_THRESHOLD_N;
  marker.visible = visibleCop;
  if (visibleCop) {
    marker.position.set(cop[0], cop[1], cop[2]);
  }
}

function updateMetrics(frame) {
  const rightCompression = derived.rightMaxCompressionMm[frame];
  const leftCompression = derived.leftMaxCompressionMm[frame];
  const rightMargin = derived.rightBottomOutMarginMm[frame];
  const leftMargin = derived.leftBottomOutMarginMm[frame];
  const rightForce = derived.rightForceY[frame];
  const leftForce = derived.leftForceY[frame];
  state.activeSide = rightForce >= leftForce ? "right" : "left";
  const activeForce = state.activeSide === "right" ? rightForce : leftForce;
  ui.activeSoleLabel.textContent = "Right";
  ui.activeSoleLabel.title =
    `Peak pressure ${formatNumber(rightCompression, 2)} MPa, ` +
    `bottom-out margin ${formatNumber(rightMargin, 1)} mm, active vertical GRF ${formatNumber(activeForce, 0)} N`;
}

function drawInstrumentation(frame) {
  drawGrfChart(frame);
  drawSoleViews(frame);
}

function buildGrfChartSeries() {
  if (!derived.task1) {
    return [
      { data: derived.rightForceY, color: "#1769aa", width: 1.8 },
      { data: derived.leftForceY, color: "#bf6b16", width: 1.8 },
    ];
  }

  const series = [];
  if (derived.task1.predicted) {
    series.push(
      { data: derived.task1.predicted.rightForceY, color: "#1769aa", width: 1.8 },
      { data: derived.task1.predicted.leftForceY, color: "#bf6b16", width: 1.8 },
    );
  }
  return series;
}

function drawGrfChart(frame) {
  const canvas = ui.grfChart;
  const dpr = Math.min(window.devicePixelRatio, 2);
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.floor(rect.width * dpr));
  const height = Math.max(1, Math.floor(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }

  const ctx = canvas.getContext("2d");
  ctx.save();
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, rect.width, rect.height);

  const series = buildGrfChartSeries();
  if (derived.rightSoleForceY && derived.leftSoleForceY) {
    series.push(
      { data: derived.rightSoleForceDisplayY ?? derived.rightSoleForceY, color: "#1769aa", dash: [5, 4], width: 1.5 },
      { data: derived.leftSoleForceDisplayY ?? derived.leftSoleForceY, color: "#bf6b16", dash: [5, 4], width: 1.5 },
    );
  }

  drawPanel(
    ctx,
    {
      title: "",
      min: 0,
      max: derived.maxForce * 1.08,
      series,
    },
    frame,
    36,
    8,
    rect.width - 44,
    rect.height - 20,
  );
  ctx.restore();
}

function drawPanel(ctx, panel, frame, x, y, width, height) {
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(x, y, width, height);
  ctx.strokeStyle = "#b7b7b7";
  ctx.strokeRect(x, y, width, height);
  ctx.font = "14px Arial, Helvetica, sans-serif";

  if (panel.title) {
    ctx.fillStyle = "#2f2f2f";
    ctx.fillText(panel.title, x, y - 5);
  }

  for (let tick = 0; tick <= 2; tick += 1) {
    const value = panel.min + ((panel.max - panel.min) * tick) / 2;
    const ty = mapValue(value, panel.min, panel.max, y + height, y);
    ctx.strokeStyle = "#e2e2e2";
    ctx.beginPath();
    ctx.moveTo(x, ty);
    ctx.lineTo(x + width, ty);
    ctx.stroke();
    ctx.fillStyle = "#666666";
    ctx.fillText(formatNumber(value, panel.max > 100 ? 0 : 1), 4, ty + 4);
  }

  panel.series.forEach((series) => {
    ctx.strokeStyle = series.color;
    ctx.lineWidth = series.width ?? 1.8;
    ctx.setLineDash(series.dash ?? []);
    ctx.beginPath();
    series.data.forEach((value, i) => {
      const px = x + (i / (manifest.frames - 1)) * width;
      const py = mapValue(value, panel.min, panel.max, y + height, y);
      if (i === 0) {
        ctx.moveTo(px, py);
      } else {
        ctx.lineTo(px, py);
      }
    });
    ctx.stroke();
    ctx.setLineDash([]);
  });

  const cursorX = x + (frame / (manifest.frames - 1)) * width;
  ctx.strokeStyle = "#222222";
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  ctx.moveTo(cursorX, y);
  ctx.lineTo(cursorX, y + height);
  ctx.stroke();
}

function drawSoleViews(frame) {
  const payload = makeSolePlotPayload("right", frame);
  drawSolePlot(ui.soleTop, payload, "top");
  drawSolePlot(ui.soleLateral, payload, "lateral");
}

function makeSolePlotPayload(side, frame) {
  const local = side === "right" ? arrays.rightSoleLocal : arrays.leftSoleLocal;
  const output = state.nnfe.outputs[side];
  const displacement = output.displacementLocal;
  const compression = output.compression;
  const facePressure = output.facePressureMpa ?? null;
  const nodeCount = manifest.sole.nodeCount;
  const displacementOffset = frame * nodeCount * 3;
  const compressionOffset = frame * nodeCount;
  const bottomY = manifest.sole.localBounds?.min?.[1] ?? -manifest.sole.thicknessM ?? -0.025;
  const projected = new Array(nodeCount);

  for (let node = 0; node < nodeCount; node += 1) {
    const p = node * 3;
    const src = displacementOffset + p;
    const c = compression[compressionOffset + node];
    projected[node] = {
      length: local[p] + displacement[src] * state.deformationScale,
      width: local[p + 2] + displacement[src + 2] * state.deformationScale,
      height: local[p + 1] - bottomY + displacement[src + 1] * state.deformationScale,
      compression: c,
    };
  }

  return {
    projected,
    facePressure,
    facePressureOffset: frame * derived.bottomFaceQuadrature.length,
    contactSummary: state.contact[side],
  };
}

function drawSolePlot(canvas, payload, mode) {
  const dpr = Math.min(window.devicePixelRatio, 2);
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.floor(rect.width * dpr));
  const height = Math.max(1, Math.floor(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }

  const ctx = canvas.getContext("2d");
  ctx.save();
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, rect.width, rect.height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, rect.width, rect.height);

  const coords = payload.projected.map((point) => ({
    x: point.length,
    y: mode === "top" ? point.width : point.height,
    compression: point.compression,
  }));
  const bounds = computePlotBounds(coords, {
    includeY: mode === "lateral" ? 0 : null,
    padXFraction: mode === "top" ? 0.025 : 0.08,
    padYFraction: mode === "top" ? 0.04 : 0.16,
    minPad: mode === "top" ? 0.004 : 0.015,
  });
  const transform = makePlotTransform(bounds, rect.width, rect.height, mode === "top" ? 6 : 12);

  if (mode === "lateral") {
    const y = transform.mapY(0);
    ctx.strokeStyle = "#606060";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(10, y);
    ctx.lineTo(rect.width - 10, y);
    ctx.stroke();
  }

  ctx.lineJoin = "round";
  const heatFaces = manifest.sole.bottomContactFaces ?? manifest.sole.quadFaces ?? manifest.sole.faces;
  for (let faceIndex = 0; faceIndex < heatFaces.length; faceIndex += 1) {
    const face = heatFaces[faceIndex];
    const faceCoords = face.map((nodeId) => coords[nodeId]);
    const avg = payload.facePressure
      ? payload.facePressure[payload.facePressureOffset + faceIndex]
      : faceCoords.reduce((sum, point) => sum + point.compression, 0) / faceCoords.length;
    ctx.beginPath();
    faceCoords.forEach((point, index) => {
      const x = transform.mapX(point.x);
      const y = transform.mapY(point.y);
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.closePath();
    ctx.fillStyle = "rgba(205, 208, 214, 0.72)";
    ctx.fill();
    if (avg > 1.0e-5) {
      ctx.fillStyle = compressionCss(avg, 0.68);
      ctx.fill();
    }
  }

  ctx.strokeStyle = "rgba(45, 48, 54, 0.24)";
  ctx.lineWidth = 0.5;
  for (const [aIndex, bIndex] of derived.quadEdges) {
    const a = coords[aIndex];
    const b = coords[bIndex];
    ctx.beginPath();
    ctx.moveTo(transform.mapX(a.x), transform.mapY(a.y));
    ctx.lineTo(transform.mapX(b.x), transform.mapY(b.y));
    ctx.stroke();
  }

  ctx.restore();
}

function writeBodyTranslation(target, targetOffset, frame, bodyIndex) {
  const sourceOffset = (frame * manifest.bodyNames.length + bodyIndex) * 3;
  target[targetOffset] = arrays.bodyTranslations[sourceOffset];
  target[targetOffset + 1] = arrays.bodyTranslations[sourceOffset + 1];
  target[targetOffset + 2] = arrays.bodyTranslations[sourceOffset + 2];
}

function setMatrixFromBody(frame, bodyIndex, matrix) {
  const bodyCount = manifest.bodyNames.length;
  const rotationOffset = (frame * bodyCount + bodyIndex) * 9;
  const translationOffset = (frame * bodyCount + bodyIndex) * 3;
  const r = arrays.bodyRotations;
  const t = arrays.bodyTranslations;
  matrix.set(
    r[rotationOffset],
    r[rotationOffset + 1],
    r[rotationOffset + 2],
    t[translationOffset],
    r[rotationOffset + 3],
    r[rotationOffset + 4],
    r[rotationOffset + 5],
    t[translationOffset + 1],
    r[rotationOffset + 6],
    r[rotationOffset + 7],
    r[rotationOffset + 8],
    t[translationOffset + 2],
    0,
    0,
    0,
    1,
  );
}

function updateGeometryMaterials() {
  geometryMeshes.forEach((mesh) => {
    mesh.visible = state.geometryOpacity > 0;
    mesh.material.opacity = state.geometryOpacity;
  });
}

function updateFloorPlane() {
  if (floorGrid) {
    floorGrid.position.y = state.floorY;
  }
  if (floorMesh) {
    floorMesh.position.y = state.floorY;
  }
}

function updateControlReadouts() {
  ui.floorReadout.value = `${formatNumber(state.floorY * 1000, 0)} mm`;
}

function updateInferenceInfo() {
  if (!ui.inferenceMs || !ui.inferenceHz || !ui.inferenceUsage || !state.nnfe) {
    return;
  }
  const hz = state.nnfe.inferenceHz ?? state.nnfe.batchInferenceHz ?? 0;
  const serialEvalMs = state.nnfe.serialEvalMs ?? (hz > 0 ? 1000 / hz : 0);
  const batchElapsedMs = state.nnfe.elapsedMs ?? 0;
  const solveCount = state.nnfe.solveCount ?? footFrameSolveCount();
  const batchHz = state.nnfe.batchInferenceHz ?? inferenceHzFromElapsed(batchElapsedMs, solveCount);
  const live = state.nnfe.live;
  const liveElapsedMs = live?.lastElapsedMs ?? null;
  const liveFrameHz = liveElapsedMs && liveElapsedMs > 0 ? 1000 / liveElapsedMs : 0;
  ui.inferenceMs.textContent = formatNumber(serialEvalMs, 2);
  ui.inferenceHz.textContent = formatNumber(hz, hz >= 100 ? 0 : 1);
  if (live?.enabled) {
    const liveText = liveElapsedMs
      ? `last two-foot call ${formatNumber(liveElapsedMs, 2)} ms ` +
        `(${formatNumber(liveFrameHz, liveFrameHz >= 100 ? 0 : 1)} replay-frame/s, ` +
        `${formatNumber(liveFrameHz * 2, liveFrameHz >= 50 ? 0 : 1)} foot-solve/s)`
      : "waiting for the first two-foot call";
    ui.inferenceUsage.textContent =
      `Live playback runs one ONNX batch per replay frame for both feet; ${liveText}. ` +
      `A cached ${solveCount}-foot-frame pass loaded in ${formatNumber(batchElapsedMs, 1)} ms ` +
      `(${formatNumber(batchHz, batchHz >= 100 ? 0 : 1)} foot-frame/s) as fallback.`;
  } else {
    ui.inferenceUsage.textContent =
      `ONNX runs when the replay loads or the floor changes: ${formatNumber(batchElapsedMs, 1)} ms ` +
      `for ${solveCount} foot-frame calls (${formatNumber(batchHz, batchHz >= 100 ? 0 : 1)} foot-frame/s). ` +
      "During playback, the app renders cached NN outputs.";
  }
}

function updateLegendVisibility() {
  if (ui.legendReference) {
    ui.legendReference.hidden = true;
  }
  if (ui.legendGaitDynamics) {
    ui.legendGaitDynamics.hidden = !derived?.task1?.predicted;
  }
}

function scheduleFloorAnalysisRecompute(delayMs = 180) {
  if (!state.nnfe?.session || !state.nnfe?.metadata) {
    return;
  }
  if (state.floorAnalysisTimer) {
    window.clearTimeout(state.floorAnalysisTimer);
  }
  const requestId = ++state.floorAnalysisRequestId;
  state.floorAnalysisStatus = "Updating sole-integrated GRF trace...";
  updateFloorTuningReadout();
  state.floorAnalysisTimer = window.setTimeout(() => {
    state.floorAnalysisTimer = null;
    recomputeFloorContactReplayForCurrentFloor(requestId).catch((error) => {
      console.error("Floor-height replay update failed", error);
      if (requestId === state.floorAnalysisRequestId) {
        state.floorAnalysisStatus = `Floor update failed: ${error?.message ?? error}`;
        updateFloorTuningReadout();
      }
    });
  }, delayMs);
}

async function recomputeFloorContactReplayForCurrentFloor(requestId = ++state.floorAnalysisRequestId) {
  const nnfe = state.nnfe;
  if (!nnfe?.session || !nnfe?.metadata) {
    return null;
  }
  const floorY = state.floorY;
  const started = performance.now();
  const outputs = await runFloorContactReplayBatch(nnfe.session, nnfe.metadata, floorY);
  const elapsedMs = performance.now() - started;
  if (state.nnfe !== nnfe || requestId !== state.floorAnalysisRequestId || Math.abs(floorY - state.floorY) > 1.0e-9) {
    return null;
  }
  applyFloorContactOutputs(nnfe, outputs, elapsedMs);
  state.floorAnalysisStatus = "";
  updateFloorTuningReadout();
  return outputs;
}

function applyFloorContactOutputs(nnfe, outputs, elapsedMs = null) {
  nnfe.outputs = outputs;
  if (elapsedMs !== null) {
    nnfe.elapsedMs = elapsedMs;
    nnfe.solveCount = footFrameSolveCount();
    nnfe.batchInferenceHz = inferenceHzFromElapsed(elapsedMs, nnfe.solveCount);
  }
  resetLiveInferenceCache(nnfe);
  updateDerivedNnfeSeries();
  updateInferenceInfo();
  state.dirtyFrame = true;
  state.dirtySoles = true;
  state.dirtyChart = true;
  updateFrame();
}

function resetLiveInferenceCache(nnfe) {
  if (!nnfe?.live) {
    return;
  }
  nnfe.live.lastKey = null;
  nnfe.live.lastFrame = null;
  nnfe.live.lastFloorY = null;
  nnfe.live.pendingKey = null;
  nnfe.live.queuedKey = null;
  nnfe.live.queuedFrame = null;
  nnfe.live.queuedFloorY = null;
}

function getSoleForceReferenceTarget() {
  if (derived?.task1?.predicted) {
    return {
      key: "gaitdynamics",
      label: "GaitDynamics prediction",
      rightForceY: derived.task1.predicted.rightForceY,
      leftForceY: derived.task1.predicted.leftForceY,
    };
  }
  return {
    key: "replay",
    label: "replay GRF",
    rightForceY: derived?.rightForceY ?? [],
    leftForceY: derived?.leftForceY ?? [],
  };
}

function updateFloorTuningReadout() {
  if (!ui.floorTuning || !derived) {
    return;
  }
  const right = derived.rightSoleForceY;
  const left = derived.leftSoleForceY;
  if (!right || !left) {
    ui.floorTuning.textContent = "Waiting for sole-contact force integration...";
    return;
  }

  ui.floorTuning.textContent =
    `Move floor offset to roughly line up the recovered sole GRF with the GaitDynamics GRF. ` +
    `Late-step spikes from the simplified sole/contact shape are filtered.`;
}

function footFrameSolveCount() {
  return Math.max(0, (manifest?.frames ?? 0) * 2);
}

function inferenceHzFromElapsed(elapsedMs, solveCount) {
  return elapsedMs > 0 ? solveCount / (elapsedMs / 1000) : 0;
}

function floorInferenceKey(frame, floorY) {
  return `${frame}:${floorY.toFixed(6)}`;
}

function resize() {
  const rect = ui.viewport.parentElement.getBoundingClientRect();
  const width = Math.max(1, rect.width);
  const height = Math.max(1, rect.height);
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  state.dirtyChart = true;
}

function vectorNorm(vector) {
  return Math.hypot(vector[0] ?? 0, vector[1] ?? 0, vector[2] ?? 0);
}

function localBoundsFromFlat(local) {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < local.length; i += 3) {
    min[0] = Math.min(min[0], local[i]);
    min[1] = Math.min(min[1], local[i + 1]);
    min[2] = Math.min(min[2], local[i + 2]);
    max[0] = Math.max(max[0], local[i]);
    max[1] = Math.max(max[1], local[i + 1]);
    max[2] = Math.max(max[2], local[i + 2]);
  }
  return { min, max };
}

function makeSoleBasisNodeSets(local) {
  const bounds = localBoundsFromFlat(local);
  const xSpan = Math.max(bounds.max[0] - bounds.min[0], 1.0e-6);
  const zSpan = Math.max(bounds.max[2] - bounds.min[2], 1.0e-6);
  const xBand = xSpan * 0.08;
  const zBand = zSpan * 0.08;
  const sets = {
    xMin: [],
    xMax: [],
    zMin: [],
    zMax: [],
  };
  for (let node = 0; node < local.length / 3; node += 1) {
    const p = node * 3;
    if (local[p] <= bounds.min[0] + xBand) {
      sets.xMin.push(node);
    }
    if (local[p] >= bounds.max[0] - xBand) {
      sets.xMax.push(node);
    }
    if (local[p + 2] <= bounds.min[2] + zBand) {
      sets.zMin.push(node);
    }
    if (local[p + 2] >= bounds.max[2] - zBand) {
      sets.zMax.push(node);
    }
  }
  return sets;
}

function makeBottomSourceNodeMap(local, bottomIds) {
  const sourceByKey = new Map();
  for (const node of bottomIds) {
    const p = node * 3;
    sourceByKey.set(`${local[p].toFixed(6)}:${local[p + 2].toFixed(6)}`, node);
  }
  const out = new Int32Array(local.length / 3);
  for (let node = 0; node < out.length; node += 1) {
    const p = node * 3;
    const key = `${local[p].toFixed(6)}:${local[p + 2].toFixed(6)}`;
    out[node] = sourceByKey.get(key) ?? node;
  }
  return out;
}

function makeBottomContactIndexMap(bottomIds, nodeCount) {
  const out = new Int32Array(nodeCount);
  out.fill(-1);
  bottomIds.forEach((nodeId, index) => {
    out[nodeId] = index;
  });
  return out;
}

function buildBottomFaceQuadrature(local, bottomFaces) {
  return bottomFaces.map((face) => {
    const nodeIds = face.slice(0, 4);
    return {
      nodeIds,
      jacobian: QUAD4_DN.map((gradients) => surfaceJacobianAtQuadPoint(local, nodeIds, gradients)),
    };
  });
}

function surfaceJacobianAtQuadPoint(local, nodeIds, gradients) {
  const tangentXi = [0, 0, 0];
  const tangentEta = [0, 0, 0];
  for (let i = 0; i < 4; i += 1) {
    const node = nodeIds[i] * 3;
    const dNdxi = gradients[i][0];
    const dNdeta = gradients[i][1];
    tangentXi[0] += dNdxi * local[node];
    tangentXi[1] += dNdxi * local[node + 1];
    tangentXi[2] += dNdxi * local[node + 2];
    tangentEta[0] += dNdeta * local[node];
    tangentEta[1] += dNdeta * local[node + 1];
    tangentEta[2] += dNdeta * local[node + 2];
  }
  const crossX = tangentXi[1] * tangentEta[2] - tangentXi[2] * tangentEta[1];
  const crossY = tangentXi[2] * tangentEta[0] - tangentXi[0] * tangentEta[2];
  const crossZ = tangentXi[0] * tangentEta[1] - tangentXi[1] * tangentEta[0];
  return Math.max(Math.hypot(crossX, crossY, crossZ), 1.0e-12);
}

function centroidFromFlat(flat, ids, frame = null) {
  const nodeCount = manifest.sole.nodeCount;
  const frameOffset = frame === null ? 0 : frame * nodeCount * 3;
  const centroid = new THREE.Vector3();
  if (!ids.length) {
    return centroid;
  }
  for (const node of ids) {
    const p = frameOffset + node * 3;
    centroid.x += flat[p];
    centroid.y += flat[p + 1];
    centroid.z += flat[p + 2];
  }
  centroid.multiplyScalar(1 / ids.length);
  return centroid;
}

function soleBasisFromNodes(local, world, frame) {
  const sets = derived.soleBasisNodes;
  const xLocal = centroidFromFlat(local, sets.xMax).sub(centroidFromFlat(local, sets.xMin));
  const zLocal = centroidFromFlat(local, sets.zMax).sub(centroidFromFlat(local, sets.zMin));
  const xWorld = centroidFromFlat(world, sets.xMax, frame).sub(centroidFromFlat(world, sets.xMin, frame));
  const zWorld = centroidFromFlat(world, sets.zMax, frame).sub(centroidFromFlat(world, sets.zMin, frame));

  if (xLocal.lengthSq() < 1.0e-12 || zLocal.lengthSq() < 1.0e-12 || xWorld.lengthSq() < 1.0e-12 || zWorld.lengthSq() < 1.0e-12) {
    return {
      x: new THREE.Vector3(1, 0, 0),
      y: new THREE.Vector3(0, 1, 0),
      z: new THREE.Vector3(0, 0, 1),
    };
  }

  const x = xWorld.normalize();
  const z = zWorld.addScaledVector(x, -zWorld.dot(x)).normalize();
  const y = new THREE.Vector3().crossVectors(z, x).normalize();
  z.crossVectors(x, y).normalize();
  return { x, y, z };
}

function parseEdgeColor(colorValue) {
  if (typeof colorValue === "string" && colorValue.startsWith("#")) {
    return new THREE.Color(colorValue);
  }
  const gray = Number(colorValue);
  if (Number.isFinite(gray)) {
    return new THREE.Color(gray, gray, gray);
  }
  return new THREE.Color(0.82, 0.78, 0.7);
}

function buildFaceEdges(faces) {
  const seen = new Set();
  const edges = [];
  for (const face of faces) {
    for (let i = 0; i < face.length; i += 1) {
      const a = face[i];
      const b = face[(i + 1) % face.length];
      const lo = Math.min(a, b);
      const hi = Math.max(a, b);
      const key = `${lo}:${hi}`;
      if (!seen.has(key)) {
        seen.add(key);
        edges.push([lo, hi]);
      }
    }
  }
  return edges;
}

function computePlotBounds(
  coords,
  {
    includeY = null,
    padXFraction = 0.08,
    padYFraction = 0.16,
    minPad = 0.015,
  } = {},
) {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const point of coords) {
    minX = Math.min(minX, point.x);
    maxX = Math.max(maxX, point.x);
    minY = Math.min(minY, point.y);
    maxY = Math.max(maxY, point.y);
  }
  if (includeY !== null) {
    minY = Math.min(minY, includeY);
    maxY = Math.max(maxY, includeY);
  }
  const padX = Math.max((maxX - minX) * padXFraction, minPad);
  const padY = Math.max((maxY - minY) * padYFraction, minPad);
  return {
    minX: minX - padX,
    maxX: maxX + padX,
    minY: minY - padY,
    maxY: maxY + padY,
  };
}

function makePlotTransform(bounds, width, height, pad = 12) {
  const dataW = Math.max(bounds.maxX - bounds.minX, 1.0e-6);
  const dataH = Math.max(bounds.maxY - bounds.minY, 1.0e-6);
  const scale = Math.min((width - pad * 2) / dataW, (height - pad * 2) / dataH);
  const centerX = 0.5 * (bounds.minX + bounds.maxX);
  const centerY = 0.5 * (bounds.minY + bounds.maxY);
  const screenCx = width * 0.5;
  const screenCy = height * 0.5;
  return {
    mapX: (x) => screenCx + (x - centerX) * scale,
    mapY: (y) => screenCy - (y - centerY) * scale,
  };
}

function compressionCss(compressionMm, alpha = 1) {
  const rgb = [0, 0, 0];
  writeCompressionColor(rgb, 0, compressionMm);
  return `rgba(${Math.round(rgb[0] * 255)}, ${Math.round(rgb[1] * 255)}, ${Math.round(rgb[2] * 255)}, ${alpha})`;
}

function writeCompressionColor(target, offset, compressionMm) {
  const t = clamp(Math.pow(clamp(compressionMm / state.colorMaxMm, 0, 1), 0.58), 0, 1);
  const stops = [
    [0.0, 0.08, 0.07, 0.13],
    [0.22, 0.08, 0.22, 0.52],
    [0.45, 0.02, 0.69, 0.78],
    [0.68, 0.96, 0.43, 0.12],
    [1.0, 1.0, 0.92, 0.38],
  ];
  let lo = stops[0];
  let hi = stops[stops.length - 1];
  for (let i = 1; i < stops.length; i += 1) {
    if (t <= stops[i][0]) {
      lo = stops[i - 1];
      hi = stops[i];
      break;
    }
  }
  const local = hi[0] === lo[0] ? 0 : (t - lo[0]) / (hi[0] - lo[0]);
  target[offset] = lerp(lo[1], hi[1], local);
  target[offset + 1] = lerp(lo[2], hi[2], local);
  target[offset + 2] = lerp(lo[3], hi[3], local);
}

function mapValue(value, min, max, outMin, outMax) {
  if (max <= min) {
    return (outMin + outMax) * 0.5;
  }
  return outMin + ((clamp(value, min, max) - min) / (max - min)) * (outMax - outMin);
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function formatNumber(value, digits) {
  return Number(value).toLocaleString(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
}
