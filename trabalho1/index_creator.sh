python3 preprocess.py

if [[ $1 == "s" ]];
then
	echo "using zet with porter stemmer"
	find colecao_stem/ -name "*.sgml" | zet -i -f rec_info_stop_trab1 --big-and-fast --stem porters
else
	echo "using zet without stemmer"
	find colecao_stem/ -name "*.sgml" | zet -i -f rec_info_trab1 --big-and-fast
fi
