import { Instance, onSnapshot, types } from "mobx-state-tree";
import { createContext, useContext } from "react";

const RootModel = types
  .model({
    digitVector: types.array(types.boolean),
  })
  .volatile((self) => ({
    currentImage: null,
  }))
  .actions((self) => ({
    toggleButtonValue(index: number) {
      self.digitVector[index] = !self.digitVector[index];
    },
  }))
  .views((self) => ({
    getDigitOneHotVector() {
      return self.digitVector.map((digit) => (digit ? 1 : 0));
    },
  }));

let initialState = RootModel.create({
  digitVector: [
    true,
    false,
    false,
    false,
    false,
    false,
    false,
    false,
    false,
    false,
  ],
});

const data = localStorage.getItem("rootState");
if (data) {
  const json = JSON.parse(data);
  if (RootModel.is(json)) {
    initialState = RootModel.create(json);
  }
}

export const rootStore = initialState;

onSnapshot(rootStore, (snapshot) => {
  console.log("Snapshot: ", snapshot);
  localStorage.setItem("rootState", JSON.stringify(snapshot));
});

export type RootInstance = Instance<typeof RootModel>;
const RootStoreContext = createContext<null | RootInstance>(null);

export const Provider = RootStoreContext.Provider;
export function useMst() {
  const store = useContext(RootStoreContext);
  if (store === null) {
    throw new Error("Store cannot be null, please add a context provider");
  }
  return store;
}
