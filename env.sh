# Source in every shell that runs the deploy stack:  source env.sh
export G1_STUDENT_DEPLOY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LUCID_DEPLOY_ROOT="${LUCID_DEPLOY_ROOT:-$HOME/lucid-sonic/lucid-g1-deploy}"
_dds="${CYCLONEDDS_HOME:-$HOME/opt/sonic-deploy-toolchain/cyclonedds}"
[ -d "$_dds/lib" ] && export CYCLONEDDS_HOME="$_dds" LD_LIBRARY_PATH="$_dds/lib:${LD_LIBRARY_PATH:-}"
export PYTHON="$G1_STUDENT_DEPLOY/.venv/bin/python"
export PYTHONPATH="$G1_STUDENT_DEPLOY:$LUCID_DEPLOY_ROOT/sdk:${PYTHONPATH:-}"
export RUNNER="$LUCID_DEPLOY_ROOT/runner/target/release/g1_deploy_onnx_ref"
unset _dds
