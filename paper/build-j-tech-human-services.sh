mkdir interact_output
pdflatex -shell-escape -output-directory=interact_output main
bibtex interact_output/main
