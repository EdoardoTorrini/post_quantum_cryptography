PORT=4242
ATTACK="two"

usage() {
  echo "$0 -p PORT [default=4242] -t [default=two]"
  echo "the other option for --type-attack [two, many]"
  exit 0
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    -p|--port)
      PORT="$2"
      shift
      shift
      ;;
    -t|--type-attack)
      ATTACK="$2"
      shift
      shift
      ;;
    -h|--help)
      usage
      shift
      ;;
    *)
      shift
      ;;
  esac
done

COMMAND="./broken_ecdsa.py"
if [[ "$ATTACK" == "many" ]]; then
  COMMAND="./broken_ecdsa.py --multi-leak=50 --leak-bits=8"
fi

SHOWCASE="[+] STARTING CHALLENGE ON 0.0.0.0 $PORT"
echo "$SHOWCASE"
export FLAG="flag{f4k3_fl4g_f0r_t3st1ng}"
socat TCP-LISTEN:"$PORT",fork EXEC:"$COMMAND"

