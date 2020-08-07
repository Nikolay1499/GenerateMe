if [[ $PWD = *\tests ]]; then
  cd ..
fi
docker build -t 131099/tests -f tests/Dockerfile .
docker run 131099/tests