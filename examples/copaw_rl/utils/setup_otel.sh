if [ -z "$OTEL_EXPORTER_OTLP_ENDPOINT" ]; then
  echo "Error: OTEL_EXPORTER_OTLP_ENDPOINT environment variable is not set."
  exit 0
elif [ -z "$OTEL_ARMS_LICENSE_KEY" ]; then
  echo "Error: OTEL_ARMS_LICENSE_KEY environment variable is not set."
  exit 0
elif [ -z "$OTEL_ARMS_PROJECT" ]; then
  echo "Error: OTEL_ARMS_PROJECT environment variable is not set."
  exit 0
elif [ -z "$OTEL_CMS_WORKSPACE" ]; then
  echo "Error: OTEL_CMS_WORKSPACE environment variable is not set."
  exit 0
fi

unset LOONGSUITE_PYTHON_SITE_BOOTSTRAP
set -x
curl -fsSL https://arms-apm-cn-hangzhou-pre.oss-cn-hangzhou.aliyuncs.com/copaw-cms-plugin/install.sh | bash -s -- \
  --x-arms-license-key "${OTEL_ARMS_LICENSE_KEY}" \
  --x-arms-project "${OTEL_ARMS_PROJECT}" \
  --x-cms-workspace "${OTEL_CMS_WORKSPACE}" \
  --serviceName "qwenpaw-agent" \
  --endpoint ${OTEL_EXPORTER_OTLP_ENDPOINT}
