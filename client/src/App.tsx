import React from "react";
import "./App.css";
import { Provider, rootStore } from "./models/RootStore";
import ToggleButton from "@material-ui/lab/ToggleButton";
import ToggleButtonGroup from "@material-ui/lab/ToggleButtonGroup";
import { observer } from "mobx-react-lite";

const App = observer(() => {
  const [image, setImage] = React.useState("");

  async function getImage() {
    let imageBlob;
    try {
      const fetchResult = await fetch(
        `http://localhost:8080/api/generate-image`,
        {
          method: "POST",
          body: JSON.stringify(rootStore.getDigitOneHotVector()),
        }
      );
      imageBlob = await fetchResult.blob();
    } catch (err) {
      return null;
    }
    setImage(URL.createObjectURL(imageBlob));
  }

  React.useEffect(() => {
    getImage();
  }, []);

  return (
    <Provider value={rootStore}>
      <div className="App">
        <header className="App-header">
          <div style={{ paddingBottom: "50px" }}>CGAN!</div>
          <div>
            <ToggleButtonGroup
              onChange={(event, val) => {
                rootStore.toggleButtonValue(val);
                getImage();
              }}
            >
              {rootStore.digitVector.map((selected, index) => {
                return (
                  <ToggleButton key={index} value={index} selected={selected}>
                    {index}
                  </ToggleButton>
                );
              })}
            </ToggleButtonGroup>
          </div>
          <div style={{ paddingTop: "100px" }}>
            <img
              style={{ height: "200px", width: "200px" }}
              src={image}
              alt={"cgan"}
            />
          </div>
        </header>
      </div>
    </Provider>
  );
});

export default App;
