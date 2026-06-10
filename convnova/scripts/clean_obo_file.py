from pathlib import Path
import pronto

input_obo = Path("../so.obo")
clean_obo = Path("../so.clean.obo")

with input_obo.open(encoding="utf-8") as fin, clean_obo.open("w", encoding="utf-8") as fout:
    for line in fin:
        # Remove the xrefs with backlash
        if (line.startswith("xref:") and "\\" in line) or (line.startswith("def:")):
            continue
        fout.write(line)

so = pronto.Ontology(str(clean_obo))