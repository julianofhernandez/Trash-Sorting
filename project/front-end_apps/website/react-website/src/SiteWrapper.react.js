import * as React from "react";
import { NavLink, withRouter } from "react-router-dom";

import { Site, Nav, Grid, Button, RouterContextProvider } from "tabler-react";

type Props = {|
  +children: React.Node,
|};

type subNavItem = {|
  +value: string,
  +to?: string,
  +icon?: string,
  +LinkComponent?: React.ElementType,
  +useExact?: boolean,
|};

type navItem = {|
  +value: string,
  +to?: string,
  +icon?: string,
  +active?: boolean,
  +LinkComponent?: React.ElementType,
  +subItems?: Array<subNavItem>,
  +useExact?: boolean,
|};

const navBarItems: Array<navItem> = [
  {
    value: "Classify",
    to: "/",
    icon: "cpu",
    LinkComponent: withRouter(NavLink),
    useExact: true,
  },
];

// const siteLogo = require("/Users/bryanburch/Desktop/logo.jpg");
class SiteWrapper extends React.Component<Props, State> {
  render(): React.Node {
    return (
      <Site.Wrapper
        headerProps={{
          href: "/",
          alt: "Trash Classifier",
          imageURL: "https://i.imgur.com/2CtSBoY.png",
          navItems: (
            <Nav.Item type="div" className="d-none d-md-flex">
              <a
                href="https://github.com/julianofhernandez/Trash-Sorting"
                target="_blank"
              >
                <Button color="white" outline className="github-btn">
                  <i className="fe fe-github mr-2" /> Source Code
                </Button>
              </a>
            </Nav.Item>
          ),
        }}
        navProps={{ itemsObjects: navBarItems }}
        routerContextComponentType={withRouter(RouterContextProvider)}
        footerProps={{
          copyright: (
            <React.Fragment>
              UI from
              <a
                href="https://github.com/tabler/tabler-react"
                target="_blank"
                rel="noopener noreferrer"
              >
                {" "}
                tabler-react
              </a>{" "}
            </React.Fragment>
          ),
          nav: (
            <React.Fragment>
              <Grid.Col auto={true}>
                <Button
                  href="https://github.com/julianofhernandez/Trash-Sorting"
                  size="sm"
                  target="_blank"
                  outline
                  color="primary"
                  RootComponent="a"
                >
                  <i className="fe fe-github mr-2" /> GitHub
                </Button>
              </Grid.Col>
            </React.Fragment>
          ),
        }}
      >
        {this.props.children}
      </Site.Wrapper>
    );
  }
}

export default SiteWrapper;
