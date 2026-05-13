PORT=4242
if [ $# -gt 0 ]; then
        PORT=${1}
fi

SHOWCASE="[+] STARTING CHALLENGE ON 0.0.0.0 $PORT"
echo "$SHOWCASE"
export FLAG="CCIT{f4k3_fl4g_f0r_t3st1ng}"
socat TCP-LISTEN:"$PORT",fork EXEC:./broken_ecdsa.py

