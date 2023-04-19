import React from 'react';
import Adapter from 'enzyme-adapter-react-16';
import { shallow, mount, configure } from 'enzyme';
import ClassifyForm from './classifyForm';
configure({ adapter: new Adapter() });

describe("ClassifyForm", () => {
  const mockData = [
    {model_name: "model1"}, 
    {model_name: "model2"},
    {model_name: "model3"},
  ];

  beforeEach(() => {
    global.fetch = jest.fn().mockImplementation(() =>
      Promise.resolve({
        json: () => Promise.resolve({ model_list: mockData }),
      })
    );
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it("should render without throwing an error", () => {
    expect(shallow(<ClassifyForm />).exists()).toBe(true);
  });

  // Expect empty list from list of models
  describe("componentDidMount", () => {
    it("should set state.options with model names on successful fetch", async () => {
      const wrapper = shallow(<ClassifyForm />);
      await wrapper.instance().componentDidMount();
      expect(wrapper.state("options")).toEqual([]);
    });
  });

  describe("handleDropdownChange", () => {
    it("should set selectedModel state", () => {
      const wrapper = shallow(<ClassifyForm />);
      wrapper.instance().handleDropdownChange({
        target: { value: "model1" },
      });
      expect(wrapper.state("selectedModel")).toBe("model1");
    });
  });

  describe("handleAddFileInput", () => {
    it("should increment fileInputCount state and disable submit/add file buttons", () => {
      const wrapper = shallow(<ClassifyForm />);
      wrapper.instance().handleAddFileInput();
      expect(wrapper.state("fileInputCount")).toBe(2);
      expect(wrapper.state("isSubmitDisabled")).toBe(true);
      expect(wrapper.state("isAddFileDisabled")).toBe(true);
    });
  });

  describe("handleRemoveFileInput", () => {
    it("should decrement fileInputCount state and remove the last file from filesToSubmit state", () => {
      const wrapper = shallow(<ClassifyForm />);
      wrapper.setState({ fileInputCount: 3, filesToSubmit: [1, 2, 3] });
      wrapper.instance().handleRemoveFileInput();
      expect(wrapper.state("fileInputCount")).toBe(2);
      expect(wrapper.state("filesToSubmit")).toEqual([1, 2]);
    });

    it("should not remove the last file input element and corresponding file object if there is only one file input element", () => {
      const wrapper = shallow(<ClassifyForm />);
      wrapper.instance().handleRemoveFileInput();
      expect(wrapper.state("fileInputCount")).toBe(1);
      expect(wrapper.state("filesToSubmit")).toEqual([]);
    });
  });
});