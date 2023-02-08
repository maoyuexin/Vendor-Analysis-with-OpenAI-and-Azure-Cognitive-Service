# vendor-invoice-analysis

End to end solution for vendor invoice analysis using Azure Cognitive Service and Azure ML. This repositiory including the source code to create the Azure pipeline and each components for the Vendor Invoice Analysis PoC.  

Invoice Data used in this PoC : https://huggingface.co/datasets/rvl_cdip

 

 ## Introduction
 
![Screenshot 2022-12-06 200734](https://user-images.githubusercontent.com/3723642/206070183-0fe4f2de-451d-4730-89d7-62ccc52f429e.png)

 ## Solution Architechture

![Screenshot 2022-12-06 200651](https://user-images.githubusercontent.com/3723642/206070050-8425a22d-c52c-4f42-9ed5-b395423553f8.png)



![Screenshot 2022-12-06 200317](https://user-images.githubusercontent.com/3723642/206069670-92e037c0-8c1b-43a1-a3e9-581da46149ca.png)

## Code Directory

|Directory|Description|
|-|-|
|clustering-src|Including the Python source file for the clustering component (Based on customized EM algorithm). |
|data-categorzation-src|Including the Python source file for the data classification component.|
|data-enrichment-src|Including the Python source file for the data enrichment component using Bing APIs.|
|data-extraction-src|Including the Python source file for the data extraction component using Azure Form Recognizer. |
|data-visualization-src|Including the Python source file for the data prepration for PowerBI visualization.|
|clustering_pipeline|Sample Yaml file to run a Azure ML job for data clustering|
|environment|Jupyter notebook to show how to create a python enviroment in Azure.|
|clustering_standalone|Sample Yaml file to run a Azure ML job for data clustering standalone mode.|
|data_categorzation|Sample Yaml file to register a data categorization component.|
|data_enrichment|Sample Yaml file to register a data enrichment component.|
|data_visualization|Sample Yaml file to register a data transformation for PowerBI visualization component.|
|pipeline_pair_clustering|Main Jupyter notebook that shows the flow to run register components and run the PoC pipeline.|


## How to Set Up

**`Notice`** : Users need a active Azure subscription to set up the related services (Azure ML, Form recognizer, Bing APIs, Cognitive Service APIs, Storage Account) and realted configuration for those Azure services to make the pipeline work. 

#### Step 1:

Create the corresponding Azure services (Azure ML, Form recognizer, Bing APIs, Cognitive Service APIs, Storage Account) and get the subscription keys.  Add the subscription keys information in the python script for each components if needed

#### Step 2:

Follow the enviroment/enviroment.ipynb to create the python envrioment to run the Azure ML jobs

#### Step 3:

Follow the pipeline_pair_clustering.ipynb to create the Azure ML reuseable components and pipeline


**`Notice`** :  the classification model is trained seperately and deployed as an endpoint in AzureML which is not included in this repository. You can skip the data categorization step

### Standalone version:

If you would like to test a seperate component or python script instead of setting up the pipeline,  you can directly run the python script and find the run command in the corresponding yaml file

e.g., the following python command is used to run the product clustering code
```
 python pair_clustering.py 
  --file_path_in ${{inputs.file_path_in}}
  --file_name ${{inputs.file_name}}
  --file_type ${{inputs.file_type}}
  --block_attributes ${{inputs.block_attributes}}
  --cluster_attributes ${{inputs.cluster_attributes}}
  --similarity_threshold ${{inputs.similarity_threshold}}
  --file_path_out ${{outputs.file_path_out}}
```  



<br />
<br />
<br />
<br />
<br />
<br />


**`Disclaimer of Warranty`** THE CODE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


