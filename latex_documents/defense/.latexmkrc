# This deck uses fontspec (Adwaita Sans) → it MUST be built with LuaLaTeX.
# LaTeX Workshop's default recipe calls `latexmk -pdf`, which would otherwise
# run pdflatex and fail with "fontspec requires XeTeX or LuaTeX".
#
# Two safeguards so the editor's build button just works:
#   * $pdf_mode = 4  → latexmk uses lualatex by default
#   * $pdflatex override → even if the recipe forces -pdf (pdflatex mode),
#     the "pdflatex" step actually runs lualatex.
$pdf_mode  = 4;
$lualatex  = 'lualatex -interaction=nonstopmode -synctex=1 -shell-escape %O %S';
$pdflatex  = 'lualatex -interaction=nonstopmode -synctex=1 -shell-escape %O %S';
