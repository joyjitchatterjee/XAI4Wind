# XAI4Wind 
<img src="https://user-images.githubusercontent.com/18656061/84653131-7b479f00-af2a-11ea-9c52-e8505f48d7b2.png" width="220" height="230">
Supplementary data for our paper "XAI4Wind: A Multimodal Knowledge Graph Database for Explainable Decision Support in Operations & Maintenance of Wind Turbines" (in submission).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
![](https://img.shields.io/static/v1?label=Neo4j&message=Cypher&color=Red)
![](https://img.shields.io/static/v1?label=Programming&message=Python&color=Green)

# Guidance on usage
The knowledge graph (KG) was exported to a JSON file using Neo4j's APOC (Awesome Procedures on Cypher) library. More details on APOC can be found here at https://neo4j.com/developer/neo4j-apoc/. This KG was developed and tested in Neo4j version 1.3.11.42 on MacOS Big Sur (version 11.0.1).
The _EXPORT JSON_ procedure was utilised to export the KG to the JSON object. Thereby, it can be loaded using its import counterpart method _apoc.import.json_.
We have also provided the Cypher script used for developing this KG in the repo as a text file _CypherScript_XAI4Wind.txt_.

An example of importing the graph through the JSON object using APOC is given below:-
```
CALL apoc.import.json("CompleteKG_XAI4Wind.json")
```

The KG in itself (standalone) can serve independently in the Neo4j Desktop Application for interactive information querying and retrieval. This does not require any availablity of SCADA datasets. 

For demonstration of XAI, SCADA features would be needed- but they can also be generalised to any resource with time-series parameters. As the KG will run in the local runtime, it is essential to connect it with a Python interface (e.g. Jupyter Notebook/Google Colaboratory) to integrate it with an Explainable AI model. This requires the _py2neo_ library which can be installed with 
```
pip install py2neo
```
or (in Colab)
```
!pip install py2neo
```
You would need to look-up the Bolt server address (as visible in your specific Neo4j Desktop Application instance) and specify the same during integrated with Python. E.g. if your Bolt server address is _11005_, and you have also specified a password for access (_pass_),the follow lines of code can be used for the interfacing.
```

from py2neo import Graph
graph = Graph("bolt://localhost:11005", auth=("neo4j", "pass")) #Can look up port address from inside Neo4j (11005 at present)
```

# Acknowledgments
We acknowledge the publicly available Skillwind maintenance manual and ORE Catapult's Platform for Operational Data (POD) for the valuable resources used in this paper.
A subset of 102 SCADA features (with their names publicly available on POD) was used in the KG. The maintenance actions segment is used to organise the information present in the Skillwind manual into a domain-specific ontology. Due to confidentiality reasons, we have not provided the numeric values and complete SCADA corpus, but only released information which is presently in the public domain in this repository. More information can be found in references (1) for the maintenance action manual and (2) for multiple other potential SCADA features and alarms available in POD.

## References
1. Skillwind Manual (https://skillwind.com/wp-content/uploads/2017/08/SKILWIND_Maintenance_1.0.pdf) 
2. Platform for Operational Data (https://pod.ore.catapult.org.uk). 

# License

This repo is based on the MIT License, which allows free use of the provided resources, subject to the original sources being credit/acknowledged appropriately. The software/resources under MIT license is provided as is, without any liability or warranty at the end of the authors.
