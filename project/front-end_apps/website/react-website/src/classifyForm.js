import * as React from "react";
import { Page, Button, Form, Alert, GalleryCard, Grid } from "tabler-react";
import { Trash, Leaf, Recycle, Biohazard } from "tabler-icons-react";

const DEV_MODELS_URL = "http://localhost:5001/read/model/list";
const DEV_BASE_READ_URL = "http://localhost:5001/read/inference/";
const DEV_BASE_READ_BATCH_URL = "http://localhost:5001/read/batch-inference/";

const MODELS_URL = "/read/model/list";
const BASE_READ_URL = "/read/inference/";
const BASE_READ_BATCH_URL = "/read/batch-inference/";

const BIN = {
  Garbage: "Garbage",
  Recyclable: "Recyclable",
  Organic: "Organic Waste",
  Hazard: "Household hazardous waste",
};

const FILE_INPUT_LIMIT = 10;

class ClassifyForm extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      options: [],
      selectedModel: "",
      fileInputCount: 1,
      filesToSubmit: [],
      submittedFiles: [],
      isSubmitDisabled: true,
      isAddFileDisabled: true,
      response: null,
      serverError: null,
    };
    this.handleDropdownChange = this.handleDropdownChange.bind(this);
    this.handleAddFileInput = this.handleAddFileInput.bind(this);
    this.handleRemoveFileInput = this.handleRemoveFileInput.bind(this);
    this.handleFileChange = this.handleFileChange.bind(this);
    this.handleFormSubmit = this.handleFormSubmit.bind(this);
    this.submitSingleFile = this.submitSingleFile.bind(this);
    this.submitMultipleFiles = this.submitMultipleFiles.bind(this);
    this.getInferenceOutput = this.getInferenceOutput.bind(this);
    this.getIconFromTextClass = this.getIconFromTextClass.bind(this);
  }

  /* Add a file input element */
  handleAddFileInput() {
    const { fileInputCount } = this.state;
    this.setState({
      fileInputCount: fileInputCount + 1,
      isSubmitDisabled: true,
      isAddFileDisabled: true,
    });
  }

  /* Remove the last added file input element and corresponding file object */
  handleRemoveFileInput() {
    const { fileInputCount, filesToSubmit } = this.state;

    if (fileInputCount > 1) {
      const newFileInputCount = fileInputCount - 1;
      let newFiles = [...filesToSubmit];
      newFiles.pop();

      this.setState({
        fileInputCount: newFileInputCount,
        filesToSubmit: newFiles,
        isSubmitDisabled: newFiles.length === newFileInputCount ? false : true,
        isAddFileDisabled: newFiles.length < newFileInputCount ? true : false,
      });
    }
  }

  /* Populate "select model" dropdown upon initial app load */
  componentDidMount() {
    fetch(MODELS_URL, {
      method: "GET",
      headers: {
        //specify which domains are allowed to make requests to server
        "Access-Control-Allow-Origin": "*", //allow all access for testing purposes
        //header to specify which HTTP methods are allowed
        "Access-Control-Allow-Methods": "GET",
        //This header specifies which headers are allowd in the request
        "Access-Control-Allow-Headers": "Content-Type",
      },
    })
      .then((response) => response.json())
      .then((data) => data["model_list"].map((item) => item.model_name))
      .then((options) => {
        this.setState({ options });
        // select first model by default
        this.setState({ selectedModel: options[0] });
      })
      .catch((error) => {
        this.setState({ serverError: error });
        console.error(error);
      });
  }

  /* Update current model */
  handleDropdownChange(event) {
    this.setState({ selectedModel: event.target.value });
  }

  /* Send one or more files to the inference server */
  handleFormSubmit(event) {
    event.preventDefault();
    const { filesToSubmit } = this.state;

    // ensure app doesn't use previous response if there is one before React finally updates it
    this.setState({ response: null });

    if (filesToSubmit.length > 1) {
      this.setState({ submittedFiles: [...filesToSubmit] });
      this.submitMultipleFiles();
    } else {
      this.setState({ submittedFiles: filesToSubmit });
      this.submitSingleFile(filesToSubmit);
    }
  }

  submitMultipleFiles() {
    const { selectedModel, filesToSubmit } = this.state;
    const formData = new FormData();
    // additional field required for batch inference
    formData.append("num_image", filesToSubmit.length);
    filesToSubmit.forEach((file, i) => formData.append(`image_${i}`, file));

    fetch(BASE_READ_BATCH_URL + selectedModel, {
      method: "POST",
      body: formData,
      headers: {
        //specify which domains are allowed to make requests to server
        "Access-Control-Allow-Origin": "*", //allow all access for testing purposes
        //header to specify which HTTP methods are allowed
        "Access-Control-Allow-Methods": "POST",
        //This header specifies which headers are allowd in the request
        "Access-Control-Allow-Headers": "Content-Type",
      },
    })
      .then((response) => response.json())
      .then((data) => {
        this.setState({
          response: data["batch_predictions"],
          serverError: null,
        });
      })
      .catch((error) => {
        this.setState({ serverError: error });
        console.error(error);
      });
  }

  submitSingleFile() {
    // console.log("submitting single file...");
    const { selectedModel, filesToSubmit } = this.state;
    const formData = new FormData();
    formData.append("image", filesToSubmit[0]);

    fetch(BASE_READ_URL + selectedModel, {
      method: "POST",
      body: formData,
      headers: {
        //specify which domains are allowed to make requests to server
        "Access-Control-Allow-Origin": "*", //allow all access for testing purposes
        //header to specify which HTTP methods are allowed
        "Access-Control-Allow-Methods": "POST",
        //This header specifies which headers are allowd in the request
        "Access-Control-Allow-Headers": "Content-Type",
      },
    })
      .then((response) => response.json())
      .then((data) => {
        this.setState({
          response: data["predictions"],
          serverError: null,
        });
      })
      .catch((error) => {
        this.setState({ serverError: error });
        console.error(error);
      });
  }

  /* Replacing or adding a new file */
  handleFileChange(event) {
    const { filesToSubmit, fileInputCount } = this.state;

    // index of file change event
    const index = parseInt(event.target.name.slice(-1));

    // append file
    if (typeof filesToSubmit[index] === "undefined") {
      this.setState({
        filesToSubmit: [...filesToSubmit, event.target.files[0]],
        isSubmitDisabled:
          filesToSubmit.length + 1 === fileInputCount ? false : true,
        isAddFileDisabled:
          filesToSubmit.length + 1 === fileInputCount &&
          filesToSubmit.length + 1 < FILE_INPUT_LIMIT
            ? false
            : true,
      });
    }
    // replace an existing file
    else {
      let newFiles = [...filesToSubmit];
      newFiles[index] = event.target.files[0];
      this.setState({ filesToSubmit: newFiles });
    }
  }

  /* Format response into rows of predictions with their corresponding image file */
  getInferenceOutput() {
    const { response, submittedFiles } = this.state;
    // might have to wrap in array if single prediction
    let _response = response;
    if (response) {
      if (!Array.isArray(response)) {
        _response = [response];
      }

      let i = _response.length - 1;
      const inferenceOutput = _response.map((prediction, i) => (
        <Grid.Row>
          <Grid.Col>
            <GalleryCard>
              <GalleryCard.Image src={URL.createObjectURL(submittedFiles[i])} />
            </GalleryCard>
          </Grid.Col>
          <Grid.Col>
            <h1>
              {this.getIconFromTextClass(prediction.trash_class)}{" "}
              {prediction.trash_class}
            </h1>
            <p>
              The most probable bin (trash_class) for what appears to be:{" "}
              {prediction.object_class}.
            </p>
            <h4>Alternative: {prediction.object_trash_class}</h4>
            <p>Based on object_trash_class.</p>
          </Grid.Col>
        </Grid.Row>
      ));
      return inferenceOutput;
    }
    return <p>Submit one or more image files for classification.</p>;
  }

  /* Get a nice icon for a bin class */
  getIconFromTextClass(predClass) {
    let icon = null;
    switch (predClass) {
      case BIN.Garbage:
        icon = <Trash size={24} strokeWidth={2} color={"black"} />;
        break;
      case BIN.Organic:
        icon = <Leaf size={24} strokeWidth={2} color={"green"} />;
        break;
      case BIN.Recyclable:
        icon = <Recycle size={24} strokeWidth={2} color={"DodgerBlue"} />;
        break;
      case BIN.Hazard:
        icon = <Biohazard size={24} strokeWidth={2} color={"orange"} />;
        break;
    }
    return icon;
  }

  render() {
    const {
      fileInputCount,
      isAddFileDisabled,
      isSubmitDisabled,
      response,
      options,
      serverError,
    } = this.state;

    // variable number of file inputs
    const fileInputs = [...Array(fileInputCount)].map((_, i) => (
      <Form.FileInput
        key={i}
        name={`fileInput${i}`}
        label={i >= 1 ? `Choose image file ${i + 1}` : "Choose an image file"}
        onChange={this.handleFileChange}
      />
    ));

    // main file form
    let fileForm = (
      <div>
        <Form.Group label="File Input">
          {fileInputs}
          <Button
            link
            onClick={this.handleAddFileInput}
            disabled={isAddFileDisabled}
          >
            Add another file
          </Button>
          {fileInputCount > 1 && (
            <Button link onClick={this.handleRemoveFileInput}>
              Remove
            </Button>
          )}
          <div className="d-flex">
            <a href="/">
              <Button link>Reset</Button>
            </a>
            <Button
              type="submit"
              color="primary"
              className="ml-auto"
              onClick={this.handleFormSubmit}
              disabled={isSubmitDisabled}
            >
              Bin it!
            </Button>
          </div>
        </Form.Group>
        <Page.Card title="Result">
          {!response && this.getInferenceOutput()}
          {response && this.getInferenceOutput()}
        </Page.Card>
      </div>
    );

    // display an alert if there's an error and prevent selecting a model
    if (serverError) {
      return (
        <div>
          <Page.Card>
            <Alert type="danger" icon="alert-triangle">
              <strong>Trouble reaching the server.</strong> Try reloading the
              page. If the problem persists try again later.
            </Alert>
          </Page.Card>
          <Page.Card>{fileForm}</Page.Card>
        </div>
      );
    }

    return (
      <Page.Card>
        <Form.Group label="Select Model">
          <Form.Select onChange={this.handleDropdownChange}>
            {options.map((option) => (
              <option key={option} value={option.value}>
                {option}
              </option>
            ))}
          </Form.Select>
        </Form.Group>
        {fileForm}
      </Page.Card>
    );
  }
}

export default ClassifyForm;
