SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
export PYTHONPATH="${SCRIPT_DIR}/ViPSAM_:${PYTHONPATH}"
python test.py --config ViPSAM_/configs/test_config.json