import * as React from "react";
import {
  Page,
  Card,
  Button,
  Form
} from "tabler-react";

class ClassifyForm extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
          fileInputCount: 1
        };
        this.handleAddFileInput = this.handleAddFileInput.bind(this);
        this.handleRemoveFileInput = this.handleRemoveFileInput.bind(this);
    }
    handleAddFileInput() {
        this.setState({
          fileInputCount: this.state.fileInputCount + 1
        });
    }
    handleRemoveFileInput() {
        const { fileInputCount } = this.state;
        if (fileInputCount > 1) {
          const newFileInputCount = fileInputCount - 1;
          this.setState({
            fileInputCount: newFileInputCount
          });
        }
    }

    render() {
        const { fileInputCount } = this.state;
        const fileInputs = [...Array(fileInputCount)].map((_, i) => (
        <Form.FileInput
            key={i}
            name={`fileInput${i}`}
            label={`Choose file ${i + 1}`}
        />
        ));
        return (
        <Page.Card>
            <Form.Group label='Select Model'>
                <Form.Select>
                    <option>Test</option>
                    <option>default</option>
                    <option>Windows</option>
                </Form.Select>
            </Form.Group>
            <Form.Group label='File Input'>
                <div>
                    {fileInputs}
                </div>
                <Button link onClick={this.handleAddFileInput} >Add another file</Button>
                {fileInputCount > 1 && (
                    <button link onClick={this.handleRemoveFileInput}>Remove</button>
                )}
            </Form.Group>
            <div className="d-flex">
            <a href="/">
                <Button link>Cancel</Button>
            </a>
            <Button type="submit" color="primary" className="ml-auto" onclick="request_classify()">
                Submit Image
            </Button>
            </div>
            <Page.Card title="Result">
                <Card.Body>
                Display result here
                </Card.Body>
            </Page.Card>
        </Page.Card>
        );
    }
}



export default ClassifyForm;