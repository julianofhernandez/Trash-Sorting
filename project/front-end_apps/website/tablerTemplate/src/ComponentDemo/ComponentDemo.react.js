// @flow

import * as React from "react";
import cn from "classnames";


type Props = {|
  +children: React.Element<any>,
  +className?: string,
  +asString?: string,
|};

type State = {|
  codeOpen: boolean,
|};

class ComponentDemo extends React.PureComponent<Props, State> {
  state = {
    codeOpen: false,
  };
  handleSourceButtonOnClick = (e: SyntheticMouseEvent<HTMLInputElement>) => {
    e.preventDefault();
    this.setState(s => ({ codeOpen: !s.codeOpen }));
  };

  render() {
    const { className, children } = this.props;
    
    const classes = cn("ComponentDemo", className);
    return (
      <div className={classes}>
        
        <div className="example">{children}</div>
      </div>
    );
  }
}

export default ComponentDemo;
