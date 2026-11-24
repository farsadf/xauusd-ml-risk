// go_parity loads an exported model exactly the way the bot's Go filter does
// (leaves.LGEnsembleFromFile(path, true) + PredictSingle on a schema-ordered vector,
// missing keys -> 0.0) and prints one probability per input row.
//
//	go run . -model ../../runs/.../m.txt -features ../../runs/.../m_features.json -input rows.json
//
// rows.json: a JSON list of {feature_name: value} maps.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/dmitryikh/leaves"
)

func main() {
	modelPath := flag.String("model", "", "LightGBM text model")
	schemaPath := flag.String("features", "", "feature schema JSON")
	inputPath := flag.String("input", "", "JSON list of feature maps")
	flag.Parse()

	model, err := leaves.LGEnsembleFromFile(*modelPath, true)
	if err != nil {
		log.Fatal(err)
	}
	var schema []string
	mustJSON(*schemaPath, &schema)
	var rows []map[string]float64
	mustJSON(*inputPath, &rows)

	for _, feats := range rows {
		v := make([]float64, len(schema))
		for i, k := range schema {
			if x, ok := feats[k]; ok {
				v[i] = x
			} // else 0.0, as in the bot
		}
		fmt.Printf("%.17g\n", model.PredictSingle(v, 0))
	}
}

func mustJSON(path string, out any) {
	b, err := os.ReadFile(path)
	if err != nil {
		log.Fatal(err)
	}
	if err := json.Unmarshal(b, out); err != nil {
		log.Fatal(err)
	}
}
