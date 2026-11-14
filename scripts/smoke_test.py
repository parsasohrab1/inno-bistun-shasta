"""Quick end-to-end smoke test for the core pipeline (not pytest)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.core.apc import APCController, ProcessState
from backend.app.core.classifier import DefectClassifier
from backend.app.core.detector import PaperDefectDetector
from backend.app.core.simulator import PaperMachineSimulator

MODEL = ROOT / "models" / "classifier.joblib"

# 1. detector + classifier
det = PaperDefectDetector()
if MODEL.exists():
    clf = DefectClassifier.load(MODEL)
    det.attach_classifier(clf)
    print("[detector] classifier attached:", clf.classes_)

# 2. APC standalone
apc = APCController()
st = ProcessState(speed=820, steam_pressure=3.6, chem_dosage=1.6, consistency=4.5,
                  defect_rate=0.06, quality_score=0.7)
act = apc.step(st)
print("[apc] in_spec:", act.in_spec, "deltas:", act.deltas)
print("[apc] energy GJ/t:", act.current_energy_gj_per_t, "->", act.predicted_energy_gj_per_t,
      " saving:", act.energy_saving_gj_per_t)
print("[apc] co2 kg/t:", act.current_co2_kg_per_t, "->", act.predicted_co2_kg_per_t,
      " saving:", act.co2_saving_kg_per_t)
print("[apc] defect_rate_reduction:", act.defect_rate_reduction,
      " tonnes_saved_per_h:", act.tonnes_saved_per_h)
for r in act.rationale[:3]:
    print("   -", r)

# 3. simulator
sim = PaperMachineSimulator(detector=det, apc=apc, seed=7)
print("\n[sim] running 8 ticks ...")
for i in range(8):
    ev = sim.tick()
    print(f"  t={ev['t']:5.1f} truth={ev['label_truth']:11s} pred={ev['detection']['pred_class']!s:>11s} "
          f"q={ev['detection']['quality_score']:.2f} n_def={ev['detection']['n_defects']} "
          f"speed={ev['state']['speed']:.0f} steam={ev['state']['steam_pressure']:.2f} "
          f"co2_saved={ev['cumulative']['co2_saved_kg']:.2f}")
print("[sim] cumulative:", sim.cum.to_dict())
print("\n[SMOKE OK]")
