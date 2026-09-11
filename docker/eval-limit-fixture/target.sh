set -eu
case "$(cat /input/mode)" in
  output) dd if=/dev/zero of=/output/overflow bs=1024 count=2048 2>/dev/null ;;
  timeout) : ;;
  *) exit 64 ;;
esac
sleep 60
