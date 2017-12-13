pdflatex -shell-escape -interaction nonstopmode main
# pdflatex -shell-escape -interaction nonstopmode main_apa6
# generate references then re-run pdf command
bibtex main
pdflatex -shell-escape -interaction nonstopmode main
