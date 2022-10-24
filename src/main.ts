import { deepCopy } from 'juandac/ase-deep-copy/src/main';
import { EyePros, FillFunction, NumTsMatrix, ProdProps } from './types';

export const NumTS = function (this: any) {
  this.__value = [];
};

const createMatrix = (shape: number | number[], fill: FillFunction = () => 0, indexes: number[] = []): NumTsMatrix[] => {
  if (Array.isArray(shape) && shape.length === 1) {
    shape = shape.pop() ?? 0;
  }

  if (typeof shape === 'number') {
    return Array.from({ length: shape }, (_, index) => fill([...indexes, index]));
  }

  const [length, ...next] = shape;
  return Array.from({ length }, (_, index) => createMatrix([...next], fill, [...indexes, index]));
};

type RandintPros = { size?: number[] };

const matrixRange =
  (generator: (shape: number[]) => number) =>
  (shape: number | [number | number[], number | number[]], { size }: RandintPros = {}): NumTsMatrix[] => {
    if (!Array.isArray(shape)) {
      shape = [0, shape];
    }

    if ((shape as number[]).length === 1) {
      const [init] = shape;
      shape = [0, init];
    }

    const [start, end] = shape as [number, number];

    if (Array.isArray(start) && Array.isArray(end)) {
      size = [end.length, start.length];

      return createMatrix(size, ([column, row]) => generator([(start as number[])[row], (end as number[])[column]]));
    }

    if (Array.isArray(start)) {
      size = [start.length];
      return createMatrix(size, ([index]) => generator([(start as number[])[index], end as number]));
    }

    if (Array.isArray(end)) {
      size = [end.length];
      return createMatrix(size, ([index]) => generator([start as number, (end as number[])[index]]));
    }

    if (!Array.isArray(size)) {
      print(`Size cannot be undefined it second parameter. Current size is undefined`);
    }

    return createMatrix(size as number[], () => generator([start as number, end as number]));
  };

NumTS.full = (shape: number | number[], to_fill = 0): NumTsMatrix[] => createMatrix(shape, () => to_fill);

NumTS.zeros = (shape: number | number[]): NumTsMatrix[] => NumTS.full(shape, 0);

NumTS.ones = (shape: number | number[]): NumTsMatrix[] => NumTS.full(shape, 1);

NumTS.rand = (shape: number | number[]): NumTsMatrix[] => createMatrix(shape, () => Math.random());

NumTS.randnintlast = (shape: number[]): number => {
  if (shape.length === 0) {
    print(`limits cannot be empty. Current limits need a value`);
  }

  if (shape.length === 1) {
    const [first] = shape;
    shape = [0, first];
  }

  const [from, to] = shape;

  const min = Math.ceil(from);
  const max = Math.floor(to);

  return Math.floor(Math.random() * (max - min + 1) + min);
};

NumTS.randnint = (shape: number[]): number => {
  if (shape.length === 0) {
    print(`limits cannot be empty. Current limits need a value`);
  }

  if (shape.length === 1) {
    const [first] = shape;
    shape = [0, first];
  }

  return Math.floor(NumTS.randn(shape));
};

NumTS.randn = (shape: number[]) => {
  if (shape.length === 0) {
    print(`limits cannot be empty. Current limits need a value`);
  }

  if (shape.length === 1) {
    const [first] = shape;
    shape = [0, first];
  }

  const [min, max] = shape;

  return Math.random() * (max - min) + min;
};

NumTS.randint = matrixRange(NumTS.randnint);
NumTS.integers = matrixRange(NumTS.randnintlast);

const isEqualsArrayWithout = (arrayOne: number[], arrayTwo: number[], without = 0, index = 0): boolean => {
  if (without !== -1) {
    arrayOne = arrayOne.filter((_, index) => index !== without);
    arrayTwo = arrayTwo.filter((_, index) => index !== without);
  }

  if (index >= arrayOne.length && index >= arrayTwo.length) return true;

  if (arrayOne[index] != arrayTwo[index]) return false;

  return isEqualsArrayWithout(arrayOne, arrayTwo, -1, index + 1);
};

NumTS.concatenate = (arrays: [NumTsMatrix[], NumTsMatrix[]] | NumTsMatrix[], axis = 0, first = true): NumTsMatrix[] => {
  if (axis < 0) print(`Axis cannot be negative. Current Axis is ${axis}`);

  const shapes = arrays.map((element) => NumTS.shape(element as NumTsMatrix[])) as [number[], number[]];
  if (first && !isEqualsArrayWithout(...shapes, axis)) print(`Concatenate cannnot with the espesific arrays because those not compatible`);

  if (axis === 0) return [...arrays.flat(1)];

  return (arrays as NumTsMatrix[][]).map((element) => NumTS.concatenate([...element], axis - 1, false));
};

NumTS.nrange = (range: number[]): NumTsMatrix[] => {
  if (range.length === 0) {
    print(`Range cannot be empty. Current range is []`);
  }

  let [start, end, advance] = range;

  if (typeof advance !== 'number') {
    advance = 1;
  }

  if (advance <= 0) {
    print(`Step cannot be negative or zero. Current step is ${advance}`);
  }

  if (range.length === 1) {
    end = start;
    start = 0;
  }

  if (`${advance}` === `${parseInt(`${advance}`)}`)
    return Array.from({ length: end }, (_, index) => index).filter((value) => value % advance == 0 && value >= start && value < end);

  return Array.from({ length: end }, (_, index) => (index == start ? index : index * advance)).filter(
    (value) => value >= start && value < end
  );
};

NumTS.linspace = (slices: number[]) => {
  if (slices.length === 0) {
    print(`Slices cannot be empty. Current Slices is []`);
  }

  let [start, stop, samples] = slices;

  if (typeof samples !== 'number') {
    samples = 50;
  }

  if (samples <= 0) {
    print(`Samples cannot be negative or zero. Current samples is ${samples}`);
  }

  if (samples == 0) {
    return [];
  }

  if (samples == 1) {
    return [start];
  }

  if (samples == 2) {
    return [start, stop];
  }

  if (slices.length === 1) {
    stop = start;
    start = 0;
  }

  const advance = (stop * 1.0 - start * 1.0) / ((samples - 1) * 1.0);

  if (advance >= 1) {
    samples += 1;
  }

  return NumTS.nrange([start, samples, advance]);
};

NumTS.eye = (row: number, { column = row, diagonal = 0 }: EyePros = {}) => {
  const matrix = NumTS.zeros([row, column]) as number[][];
  matrix.forEach((row: number[], index) => {
    if (index + diagonal < row.length) row[index + diagonal] = 1;
  });
  return matrix;
};

// TODO: "start:end:advance"
// TODO: ":end:advance"
// TODO: "start::advance"
// TODO: "start:end:"
// TODO: add negative to indes with a funtion
// TODO: add multi dimencional slides example "1:, 0:2"
NumTS.at = (array: NumTsMatrix[], indexes: number[] | string): NumTsMatrix => {
  if (indexes.length <= 0) return -1;

  if (typeof indexes === 'string') {
    const [start, end, step, steps] = [...indexes.split(':'), array.length, 1].map((e) => Number(e));
    if (indexes.length === 1) {
      return [...array];
    }
    if (indexes.length === 2) {
      return indexes.split('').pop() === ':'
        ? array.filter((_, index) => index >= start && index < step)
        : array.filter((_, index) => index >= start && index < end);
    }
    if (indexes.length === 3) {
      return end !== 0 ? array.filter((_, index) => index >= start && index < end) : array.filter((_, index) => index % step === 0);
    }

    return 0;
  }

  const [index, ...next] = indexes;
  if (index >= array.length) {
    return -1;
  }

  if (next?.length > 0) {
    return NumTS.at(array[index] as NumTsMatrix[], next);
  }

  return array[index];
};

NumTS.shape = (array: NumTsMatrix[]): number[] => {
  if (!Array.isArray(array)) return [];
  return [array.length, ...NumTS.shape(array[0] as NumTsMatrix[])];
};

const reshape = (array: NumTsMatrix[], shape: number[], indexes: number[]): NumTsMatrix[] => {
  let index = 0;
  return createMatrix(shape, () => indexes[index++]);
};

NumTS.vector2matriz = (vector: number[], shape: number[], index = 0, advance: number[] = []): NumTsMatrix[] => {
  advance = advance.length > 0 ? advance : shape;
  const [add] = advance;
  const [length, ...next] = shape;
  if (typeof length !== 'number') return [];

  return Array.from({ length }, (_, idx) => {
    if (next.length == 0) {
      const current = index;
      index += add;
      return vector[current];
    }

    return NumTS.vector2matriz(vector, next, idx, advance);
  });
};

NumTS.reshape = (array: NumTsMatrix[], shape: number[], type: 'C' | 'F' | 'T' = 'C'): NumTsMatrix[] => {
  const oldShpae = NumTS.shape(array);
  if (NumTS.prod(oldShpae) !== NumTS.prod(shape)) {
    print('ERROR: The new shape should be compatible with the original shape.');
    debug.traceback(`ERROR: The new shape should be compatible with the original shape.`);
    return [];
  }
  if (type === 'C') {
    return reshape(array, shape, array.flat(20) as number[]);
  }
  if (type === 'T') {
    const arrayFlat = NumTS.T(array).flat(20) as number[];
    return reshape(array, shape, arrayFlat);
  }
  const arrayFlatLikeFortran = NumTS.T(array).flat(20) as number[];
  return NumTS.vector2matriz(arrayFlatLikeFortran, shape);
};

const reductionOperation =
  (operation: (this: any, acum: number, val: number) => number, init = 0) =>
  (array: NumTsMatrix[], { axis = 0, initial = init, where = [], flat = true }: ProdProps = {}): NumTsMatrix => {
    if (axis < 0) {
      print(`ERROR: Axis cannot be negative. Current Axis is ${axis} in NumTS.prod`);
      debug.traceback(`ERROR: Axis cannot be negative. Current Axis is ${axis} in NumTS.prod`);
      return [];
    }

    if ((where.length !== 0 && axis === 0, where.length !== array.length)) {
      where = [...where, ...Array.from({ length: Math.abs(array.length - where.length) }, () => true)];
    }

    if (where.length === 0 && axis === 0) {
      where = array?.map(() => true);
    }

    if (axis == 0) {
      return (array as number[])
        .map((value, index) => {
          if (!where[index]) return 1;
          if (Array.isArray(value) && !flat) return 1;
          if (Array.isArray(value) && flat) value = NumTS.prod(value) as number;
          return value;
        })
        .reduce(operation, initial) as number;
    }

    return array.map((element) => NumTS.prod(element as NumTsMatrix[], { axis: axis - 1, initial, flat, where }));
  };

NumTS.prod = reductionOperation((acum, val) => acum * val, 1);

NumTS.sum = reductionOperation((acum, val) => acum + val);

NumTS.diference = reductionOperation((acum, val) => acum - val);

NumTS.divition = reductionOperation((acum, val) => acum / val, 1);

NumTS.getColumns = (array: NumTsMatrix[], index = 0): NumTsMatrix[] => {
  const newData = [] as NumTsMatrix[];
  let counter = 0;
  const innerGetColumns = (inner_data: NumTsMatrix[], i: number) => {
    if (typeof inner_data == 'number' && i == index) {
      newData[counter++] = inner_data;
    }

    if (Array.isArray(inner_data))
      for (let j = 0; j < inner_data.length; j++) {
        innerGetColumns(inner_data[j] as NumTsMatrix[], j);
      }
  };

  innerGetColumns(array, index);
  return newData;
};

NumTS.formPattern = (data: NumTsMatrix[]) => {
  let lastAccessed = 0;
  let counter = -1;
  const temp = NumTS.getColumns(data);
  let traverse = 0;
  const yak = [];
  const dim = NumTS.shape(data);
  const rows = dim[dim.length - 2];

  for (let k = 0; k < temp.length; k++) {
    yak[++counter] = traverse;
    traverse = traverse + rows;
    if (traverse >= temp.length) {
      traverse = lastAccessed + 1;
      lastAccessed++;
    }
  }
  return yak;
};

NumTS.stackColumns = (data: NumTsMatrix[]) => {
  const dim = NumTS.shape(data);
  const columns = dim[dim.length - 1];
  const accessPattern = NumTS.formPattern(data);
  let newData: NumTsMatrix[] = [];
  for (let i = 0; i < columns; i++) {
    const yak = [];
    let counter = -1;
    const temp = NumTS.getColumns(data, i);
    for (const v of accessPattern) {
      yak[++counter] = temp[v];
    }
    newData = newData.concat(yak);
  }
  return newData;
};

NumTS.T = (array: NumTsMatrix[]): NumTsMatrix[] => {
  const finalData = NumTS.stackColumns(array);
  const dim = NumTS.shape(array).reverse();
  return NumTS.reshape(finalData, dim);
};

NumTS.isSquare = (data: NumTsMatrix[]) => {
  const dim = NumTS.shape(data);
  const rows = dim[dim.length - 2];
  const columns = dim[dim.length - 1];
  if (rows == columns && dim.length == 2) {
    return true;
  } else {
    return false;
  }
};

NumTS.twoCrossTwoDeterminant = (data: [number[], number[]]): number => {
  return data[0][0] * data[1][1] - data[0][1] * data[1][0];
};

NumTS.findDeterminant = (data: NumTsMatrix[]): number => {
  function nCrossNDeterminant(data: NumTsMatrix[]): number {
    const dim = NumTS.shape(data);
    const rows = dim[dim.length - 2];
    const column = dim[dim.length - 1];
    if (rows == 2 && column == 2) {
      return NumTS.twoCrossTwoDeterminant(data as [number[], number[]]);
    }
    const toOperate = data[0];
    const timepass: never[] = [];
    let ans = 0;
    let sign = 1;
    (toOperate as number[])?.forEach((_, i) => {
      const newData: NumTsMatrix = [];
      let tig = -1;
      for (let j = 1; j < data.length; j++) {
        const temp: NumTsMatrix = [];
        let counter = -1;
        const yak = data[j];
        (yak as number[][])?.forEach((_, v) => {
          temp[++counter] = (yak as number[][])[v] as number[];
        });
        newData[++tig] = temp;
      }
      ans = ans + sign * ((toOperate as number[])[i] as number) * nCrossNDeterminant(newData as NumTsMatrix[]);
      sign = sign * -1;
      (timepass as NumTsMatrix[])[i] = newData;
    });
    return ans;
  }

  const dim = NumTS.shape(data);
  const rows = dim[dim.length - 2];
  const columns = dim[dim.length - 1];

  if (rows != columns) {
    print('To calculate Determinant the matrix should be square');
    return 1;
  } else {
    const res: NumTsMatrix[] = [];
    let counter = -1;

    const innerDeterminant = (innerData: NumTsMatrix[]) => {
      if (NumTS.isSquare(innerData as NumTsMatrix[])) {
        res[++counter] = nCrossNDeterminant(innerData);
      } else {
        innerData.forEach((_, i) => {
          innerDeterminant(innerData[i] as NumTsMatrix[]);
        });
      }
    };

    innerDeterminant(data);

    if (res.length == 1) {
      return res[0] as number;
    } else {
      const te = dim.slice(0, dim.length - 2);
      if (te.length >= 2) {
        return NumTS.reshape(res, te) as unknown as number;
      } else {
        return res as unknown as number;
      }
    }
  }
};

NumTS.generateIdentityMatrix = (dim: number[]) => {
  const rowsNumber = dim[dim.length - 2];
  const colNumber = dim[dim.length - 1];

  if (rowsNumber != colNumber) {
    throw new Error('The dimensions should be square dimension');
  } else {
    if (dim.length == 2) {
      return NumTS.eye(rowsNumber, { column: colNumber });
    } else {
      // it is a higher dimension matrix
      const res: NumTsMatrix[] = [];
      const subset = dim.slice(0, dim.length - 2);
      for (let i = 0; i < subset.length; i++) {
        const temp = [];
        for (let j = 0; j < subset[subset.length - 1 - i]; j++) {
          if (i == 0) temp[j] = NumTS.eye(rowsNumber, { column: colNumber });
        }
        if (i == 0) {
          res[i] = temp;
        } else {
          res[i] = deepCopy(res[i - 1]);
        }
      }
      if (res.length == 1) {
        return res[0];
      } else {
        return res;
      }
    }
  }
};

NumTS.findAdjoint = (data: NumTsMatrix[]) => {
  const ansToSave: NumTsMatrix[] = [];
  let sign = 1;

  data.forEach((_, k) => {
    const toOperate = deepCopy(data[k]);
    for (const i in toOperate) {
      sign = Math.pow(-1, Number(k) + Number(i));
      const newData: NumTsMatrix[] = [];
      let tig = -1;
      for (let j = 0; j < data.length; j++) {
        if (j !== (k as unknown as number)) {
          const temp: NumTsMatrix[] = [];
          let counter = -1;
          const yak = data[j] as NumTsMatrix[];
          yak.forEach((_, v) => {
            if (v != (i as unknown as number)) {
              temp[++counter] = yak[v];
            }
          });
          newData[++tig] = temp;
        }
      }
      toOperate[i] = sign * (NumTS.findDeterminant(newData as NumTsMatrix[]) as number);
    }
    ansToSave[k] = toOperate;
  });
  return NumTS.T(ansToSave);
};

/* NumTS.matrixInverse = (data: NumTsMatrix[]) => {
  function findInverse(data: NumTsMatrix[]) {
    if (NumTS.findDeterminant(data) == 0) {
      throw new Error('Determinant has to be non Zero');
    }
    const res = NumTS.ElementMultiply(NumTS.findAdjoint(data), 1 / NumTS.findDeterminant(data));
    const dim = NumTS.shape(res);
    const rows = dim[dim.length - 2];
    const columns = dim[dim.length - 1];

    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < columns; j++) {
        // eslint-disable-next-line no-compare-neg-zero
        if (!(res as number[][])[i][j]) {
          res[i][j] = 0;
        }
      }
    }
    return res;
  }

  function innerMatrixInverse(data: NumTsMatrix[]) {
    if (NumTS.isSquare(data)) {
      res[++counter] = findInverse(data);
    } else {
      for (const i in data) {
        innerMatrixInverse(data[i] as NumTsMatrix[]);
      }
    }
  }

  const res: NumTsMatrix[] = [];
  let counter = -1;
  innerMatrixInverse(data);
  if (res.length == 1) {
    return res[0];
  } else {
    return res;
  }
};

 */

NumTS.isAllSameDimensions = (dimFirstMatrix: number[], dimSecondMatrix: number[]) => {
  if (dimFirstMatrix.length != dimSecondMatrix.length) {
    return false;
  } else {
    const temp1 = dimFirstMatrix.slice(0, dimFirstMatrix.length - 2);
    const temp2 = dimSecondMatrix.slice(0, dimSecondMatrix.length - 2);
    if (temp1.length >= 1 && temp2.length >= 1) {
      for (let i = 0; i < temp1.length; i++) {
        if (temp1[i] != temp2[i]) {
          return false;
        }
      }
      return true;
    } else {
      return true;
    }
  }
};

NumTS.matrixMultiply = (firstMatrix: NumTsMatrix[], secondMatrix: NumTsMatrix[]) => {
  let counter = -1;
  function innerMatrixMultiply(
    firstMatrix: NumTsMatrix[],
    secondMatrix: NumTsMatrix[],
    shuffle: true | false | 'allSame',
    res: NumTsMatrix[] = []
  ) {
    if (shuffle == false) {
      if (NumTS.shape(firstMatrix).length == 2) {
        res[++counter] = findMatrixMultiply(firstMatrix, secondMatrix);
      } else {
        secondMatrix.forEach((_, j) => {
          innerMatrixMultiply(firstMatrix[j] as NumTsMatrix[], secondMatrix, shuffle, res);
        });
      }
    } else if (shuffle == true) {
      if (NumTS.shape(secondMatrix).length == 2) {
        res[++counter] = findMatrixMultiply(firstMatrix, secondMatrix);
      } else {
        secondMatrix.forEach((_, j) => {
          innerMatrixMultiply(firstMatrix, secondMatrix[j] as NumTsMatrix[], shuffle, res);
        });
      }
    } else if (shuffle == 'allSame') {
      if (NumTS.shape(secondMatrix).length == 2) {
        res[++counter] = findMatrixMultiply(firstMatrix, secondMatrix);
      } else {
        secondMatrix.forEach((_, j) => {
          innerMatrixMultiply(firstMatrix[j] as NumTsMatrix[], secondMatrix[j] as NumTsMatrix[], shuffle, res);
        });
      }
    } else {
      print(`ERROR: The input ndArray has improper shapes`);
    }
  }

  function findMatrixMultiply(firstMatrix: NumTsMatrix[], secondMatrix: NumTsMatrix[]) {
    const dimFirstMatrix = NumTS.shape(firstMatrix);
    const dimSecondMatrix = NumTS.shape(secondMatrix);

    const rowsFirstMatrix = dimFirstMatrix[dimFirstMatrix.length - 2];
    const rowsSecondMatrix = dimSecondMatrix[dimSecondMatrix.length - 2];

    const columnsFirstMatrix = dimFirstMatrix[dimFirstMatrix.length - 1];
    const columnsSecondMatrix = dimSecondMatrix[dimSecondMatrix.length - 1];

    if (columnsFirstMatrix != rowsSecondMatrix) {
      print(
        `ERROR: shapes (${rowsFirstMatrix},${columnsFirstMatrix}) and (${rowsSecondMatrix},${columnsSecondMatrix}) not aligned: ${columnsFirstMatrix} (Matrix 1) != ${rowsSecondMatrix} (Matrix 2)`
      );
      return 0;
    }
    //code for Multiplication
    const res = [];
    let count = -1;
    for (let i = 0; i < rowsFirstMatrix; i++) {
      const temp = [];
      let counter = -1;
      for (let k = 0; k < columnsSecondMatrix; k++) {
        let sum = 0;
        for (let j = 0; j < rowsSecondMatrix; j++) {
          sum = sum + (firstMatrix as number[][])[i][j] * (secondMatrix as number[][])[j][k];
        }
        temp[++counter] = sum;
      }
      res[++count] = temp;
    }
    return res;
  }

  const dimFirstMatrix = NumTS.shape(firstMatrix);
  const dimSecondMatrix = NumTS.shape(secondMatrix);

  let shuffle: false | true | 'allSame' = false;
  if (dimFirstMatrix.length > dimSecondMatrix.length) {
    shuffle = false;
  } else if (dimFirstMatrix.length < dimSecondMatrix.length) {
    shuffle = true;
  } else if (NumTS.isAllSameDimensions(dimFirstMatrix, dimSecondMatrix)) {
    shuffle = 'allSame';
  }

  const res: NumTsMatrix[] = [];

  innerMatrixMultiply(firstMatrix, secondMatrix, shuffle, res);

  if (res.length == 1) {
    return res[0];
  }
  return res;
};

NumTS.editData = (data: NumTsMatrix[]) => {
  const temp = [];
  let counter = -1;
  for (let i = 0; i < data.length; i++) {
    temp[++counter] = Array.isArray(data[i]) ? ((data[i] as NumTsMatrix[]).flat(20) as number[]) : data[i];
  }
  counter = -1;
  const newData = [];
  for (let j = 0; j < (temp[0] as number[]).length; j++) {
    let sum = 0;
    for (let k = 0; k < temp.length; k++) {
      sum += (temp as number[][])[k][j];
    }
    newData[++counter] = sum / temp.length;
  }
  return newData;
};

NumTS.getProperDim = (newDim: number[]) => {
  if (newDim.length == 1) {
    newDim = [1, newDim[0]];
  }
  return newDim;
};

NumTS.meanAlongAxis = (data: NumTsMatrix[], axis: number) => {
  const dim = NumTS.shape(data);
  if (dim.length - 1 < axis) {
    print(`ERROR: value ${axis} of axis is out of bond`);
    return 0;
  }
  if (dim.length - 1 == axis) {
    let temp = data.flat(20);
    temp = NumTS.addStepWise(temp, dim[axis]);
    let newDim = dim.slice(0, dim.length - 1);
    if (newDim.length == 1) {
      newDim = [1, newDim[0]];
    }
    const toReturn = NumTS.reshape(temp, newDim);
    if (toReturn.length == 1) {
      return toReturn[0] as number;
    }
    return toReturn;
  } else if (axis == 0) {
    const newData = NumTS.reshape(NumTS.editData(data), NumTS.getProperDim(dim.slice(1)));
    if (newData.length == 1) {
      return newData[0] as number;
    } else {
      return newData;
    }
  }
  const requiredDimension = NumTS.leaveAElement(dim, axis) as number[];
  const toMakeDimension = dim.slice(axis);
  const ansFinal = NumTS.tempGetRequiredData(data, toMakeDimension, dim[axis]);
  return NumTS.reshape(ansFinal.flat(20), requiredDimension as number[]);
};

NumTS.dimensionCheck = (dim1: NumTsMatrix[], dim2: NumTsMatrix[]) => {
  for (let i = 0; i < dim1.length; i++) {
    if (dim1[i] != dim2[i]) {
      return false;
    }
  }
  return true;
};

NumTS.tempGetRequiredData = (data: NumTsMatrix[], requiredDimension: number[], step: number) => {
  const newData: NumTsMatrix[] = [];
  let counter = -1;

  function getRequiredData(data: NumTsMatrix[], requiredDimension: number[], step: number) {
    if (NumTS.dimensionCheck(NumTS.shape(data), requiredDimension)) {
      const temp = NumTS.getProperDim(requiredDimension.slice(1));
      const transposedData = NumTS.T(data);
      const flattenedData = transposedData.flat(20);
      const dataAddedStepWise = NumTS.addStepWise(flattenedData, step);
      const reshapedData = NumTS.reshape(dataAddedStepWise, temp);
      const asdf = NumTS.T(reshapedData).flat(20);
      return asdf;
      // var asdf = Miscellaneous.flatten(matrixOperations.transpose(Miscellaneous.reshape(addStepWise(Miscellaneous.flatten(matrixOperations.transpose(data)), step),getProperDim(requiredDimension.slice(1)))));
    } else {
      for (let i = 0; i < data.length; i++) {
        newData[++counter] = getRequiredData(data[i] as NumTsMatrix[], requiredDimension, step) as NumTsMatrix;
      }
    }
  }
  getRequiredData(data, requiredDimension, step);
  return newData;
};

NumTS.addStepWise = (finalData: NumTsMatrix[], steps: number) => {
  const k = [];
  let counter = -1;
  for (let i = 0; i < finalData.length; i = i + steps) {
    let sum = 0;
    for (let j = 0; j < steps; j++) {
      sum += finalData[i + j] as number;
    }
    k[++counter] = sum / steps;
  }
  return k;
};

NumTS.leaveAElement = (data: NumTsMatrix[], left: number) => {
  const k = [];
  let counter = -1;
  for (let i = 0; i < data.length; i++) {
    if (i != left) {
      k[++counter] = data[i];
    }
  }
  return k;
};

NumTS.isSingleArray = (data: NumTsMatrix[]) => {
  if (NumTS.shape(data).length == 2 && NumTS.shape(data)[0] == 1) {
    return true;
  }
  return false;
};

NumTS.hasSingleItem = (data: NumTsMatrix[]) => {
  if (data.length == 1) {
    return true;
  } else {
    return false;
  }
};

NumTS.all_dimensions_same = (firstArray: NumTsMatrix[], secondArray: NumTsMatrix[]) => {
  const firstSize = NumTS.shape(firstArray);
  const secondSize = NumTS.shape(secondArray);
  if (firstSize.length != secondSize.length) {
    return false;
  } else {
    for (let i = 0; i < firstArray.length; i++) {
      if (firstSize[i] != secondSize[i]) {
        return false;
      }
    }
    return true;
  }
};

NumTS.is_first_greater = (first_array: NumTsMatrix[], second_array: NumTsMatrix[]) => {
  const first_array_dimension = NumTS.shape(first_array);
  const second_array_dimension = NumTS.shape(second_array);
  if (first_array_dimension.length > second_array_dimension.length) {
    return true;
  } else if (first_array_dimension.length == second_array_dimension.length) {
    first_array_dimension.forEach((_, j) => {
      if (second_array_dimension[j] < first_array_dimension[j]) {
        return true;
      } else if (second_array_dimension[j] > first_array_dimension[j]) {
        return false;
      }
    });
    return true;
  } else {
    return false;
  }
};

NumTS.isPrimitive = (test: any) => test !== Object(test);

NumTS.findHighestElement = (data: NumTsMatrix[]): number => {
  function findHighest(data: NumTsMatrix[]): number {
    let max: number = data[0] as number;
    for (const v of data) {
      if (v > max) {
        max = v as number;
      }
    }
    return max;
  }
  if (NumTS.isPrimitive(data[0])) {
    return findHighest(data);
  } else {
    const max: NumTsMatrix[] = [];
    data.forEach((_, i) => {
      max[i] = NumTS.findHighestElement(data[i] as NumTsMatrix[]);
    });
    return findHighest(max as NumTsMatrix[]);
  }
};

NumTS.findLowestElement = (data: NumTsMatrix[]) => {
  function findLowest(data: NumTsMatrix[]): number {
    let min: number = data[0] as number;
    for (const v of data) {
      if (v < min) {
        min = v as number;
      }
    }
    return min;
  }
  if (NumTS.isPrimitive(data[0])) {
    return findLowest(data);
  } else {
    const max: NumTsMatrix[] = [];
    data.forEach((_, i) => {
      max[i] = NumTS.findLowestElement(data[i] as NumTsMatrix[]);
    });
    return findLowest(max);
  }
};

NumTS.findSquare = (data: NumTsMatrix[]) => {
  function inner_findSquare(inner_data: NumTsMatrix[]) {
    if (typeof inner_data == 'number') {
      return inner_data * inner_data;
    } else {
      data.forEach((_, i) => {
        inner_data[i] = inner_findSquare(inner_data[i] as NumTsMatrix[]);
      });
    }
    return inner_data;
  }
  const saf = deepCopy(data);
  return inner_findSquare(saf);
};

NumTS.findRange = (data: NumTsMatrix[]) => {
  return NumTS.findHighestElement(data) - NumTS.findLowestElement(data);
};

NumTS.findTotalElements = (data: NumTsMatrix[]): number => {
  if (typeof data[0] == 'number') {
    return data.length;
  }
  let sum = 0;

  data.forEach((_, i) => {
    sum = sum + NumTS.findTotalElements(data[i] as NumTsMatrix[]);
  });
  return sum;
};

NumTS.findMean = (data: NumTsMatrix[], axis?: number): number => {
  if (axis == undefined || axis == null) {
    const sum = NumTS.findSum(data) as number;
    const totalElements = NumTS.findTotalElements(data);
    return sum / totalElements;
  }
  return NumTS.meanAlongAxis(data, axis) as unknown as number;
};

NumTS.findSum = (data: NumTsMatrix[]): number => {
  if (typeof data == 'number') {
    return data;
  }
  let sum = 0;
  data.forEach((_, i) => {
    sum = (sum + NumTS.findSum(data[i] as NumTsMatrix[])) as number;
  });
  return sum;
};

NumTS.findMedian = (data: NumTsMatrix[]) => {
  let newdata = data.flat(20);
  newdata = newdata.slice();
  // newdata = newdata.sort();
  const values: number[] = newdata as number[];
  values.sort((a: number, b: number) => a - b);

  const half = Math.floor(values.length / 2);

  if (values.length % 2 !== 0) return values[half];
  else return (values[half - 1] + values[half]) / 2.0;
};

NumTS.findFrequency = (data: NumTsMatrix[], tofind: number): number => {
  if (typeof data == 'number') {
    if (data == tofind) {
      return 1;
    } else {
      return 0;
    }
  }
  let sum = 0;
  data.forEach((_, i) => (sum = sum + NumTS.findFrequency(data[i] as NumTsMatrix[], tofind)));
  return sum;
};

NumTS.executeOnTwoArray = (data_array: NumTsMatrix[], to_operation: NumTsMatrix[], task_to_perform: (a: number, b: number) => number) => {
  data_array.forEach((_, i) => {
    if (typeof data_array[i] === 'number') {
      data_array[i] = task_to_perform(data_array[i] as number, to_operation[i] as number);
    } else {
      NumTS.executeOnTwoArray(data_array[i] as NumTsMatrix[], to_operation[i] as number[], task_to_perform);
    }
  });

  return data_array;
};

NumTS.executeOnNumberAndArray = (
  data_array: NumTsMatrix[],
  to_operation: NumTsMatrix[],
  task_to_perform: (a: number, b: number) => number
) => {
  data_array.forEach((_, i) => {
    if (typeof data_array[i] == 'object') {
      data_array[i] = NumTS.executeOnNumberAndArray(data_array[i] as NumTsMatrix[], to_operation, task_to_perform);
    } else {
      data_array[i] = task_to_perform(data_array[i] as number, to_operation[0] as number);
    }
  });

  return data_array;
};

NumTS.executeOnNonEqualArray = (
  data_array: NumTsMatrix[],
  to_operation: NumTsMatrix[],
  task_to_perform: (a: number, b: number) => number,
  res: any[]
) => {
  const data_dimension = NumTS.shape(data_array);
  const to_operation_dimension = NumTS.shape(to_operation);
  const subset_data_dimension = data_dimension.slice(data_dimension.length - to_operation_dimension.length);

  function isInnerDimensionSame() {
    to_operation_dimension.forEach((_, i) => {
      if (to_operation_dimension[i] != subset_data_dimension[i]) {
        return false;
      }
    });
    return true;
  }

  if (isInnerDimensionSame()) {
    res.forEach(() => res.pop());
    NumTS.executeOnInnerDimensions(data_array, to_operation, data_array.slice(), 0, task_to_perform).forEach((value) => {
      res.push(value);
    });
  } else {
    if (data_dimension[data_dimension.length - 2] == to_operation_dimension[0]) {
      NumTS.executeOnColumns(data_array, to_operation, 0, task_to_perform);
    } else if (data_dimension[data_dimension.length - 1] == to_operation_dimension[1]) {
      NumTS.executeOnRow(data_array, to_operation, 0, task_to_perform);
    } else {
      throw new Error('Cannot compute the request Sorry');
    }
  }
  return data_array;
};

NumTS.executeOnColumns = (
  data_array: NumTsMatrix[],
  to_operation: NumTsMatrix[],
  i = 0,
  task_to_perform: (a: number, b: number) => number
) => {
  if (typeof data_array[0] == 'number') {
    data_array = NumTS.executeOnNumberAndArray(data_array, to_operation[i] as NumTsMatrix[], task_to_perform);
  } else if (Array.isArray(data_array[i])) {
    data_array.forEach((_, y) =>
      NumTS.executeOnColumns(data_array[y] as NumTsMatrix[], to_operation, y as unknown as number, task_to_perform)
    );
  }
};

NumTS.executeOnRow = (data_array: NumTsMatrix[], to_operation: NumTsMatrix[], i = 0, task_to_perform: (a: number, b: number) => number) => {
  // if (miscellaneousOperations.get_Dimensions(data_array).length == 2 && miscellaneousOperations.get_Dimensions(data_array)[0] == 1) {
  if (NumTS.isSingleArray(data_array)) {
    data_array.forEach((_, j) => (data_array[j] = task_to_perform(data_array[j] as number, to_operation[j] as number)));
  } else {
    data_array.forEach((_, k) => NumTS.executeOnRow(data_array[k] as NumTsMatrix[], to_operation, k as unknown as number, task_to_perform));
  }
};

NumTS.executeOnInnerDimensions = (
  data_array: NumTsMatrix[],
  to_operation: NumTsMatrix[],
  to_store = data_array.slice(),
  i = 0,
  task_to_perform: (a: number, b: number) => number
) => {
  /*
   * Here to_store is required because it task_to_perform with it self and creates a loop
   * so to avoid circular we need a separate array
   */
  if (i < data_array.length) {
    if (NumTS.all_dimensions_same(data_array, to_operation)) {
      to_store[i] = NumTS.executeOnTwoArray(data_array, to_operation, task_to_perform);
    } else {
      data_array.forEach((_, j) => {
        NumTS.executeOnInnerDimensions(
          data_array[j] as NumTsMatrix[],
          to_operation,
          (data_array[j] as NumTsMatrix[]).slice(),
          j as unknown as number,
          task_to_perform
        );
      });
    }
  }
  return to_store;
};

NumTS.innerExecute = (
  data_array: NumTsMatrix[],
  to_operation: NumTsMatrix[],
  task_to_perform: (a: number, b: number) => number,
  res: any[]
) => {
  if (NumTS.hasSingleItem(to_operation)) {
    return NumTS.executeOnNumberAndArray(data_array, to_operation, task_to_perform);
  } else if (typeof to_operation == 'object') {
    if (NumTS.all_dimensions_same(data_array, to_operation)) {
      return NumTS.executeOnTwoArray(data_array, to_operation, task_to_perform);
    } else {
      return NumTS.executeOnNonEqualArray(data_array, to_operation, task_to_perform, res);
    }
  }
  return data_array;
};

NumTS.mainExecute = (
  data_array: NumTsMatrix[],
  to_operation: NumTsMatrix[],
  replace: boolean,
  task_to_perform: (shuffle: boolean) => (a: number, b: number) => number
) => {
  if (typeof data_array == 'number') {
    const temp = data_array;
    data_array = [];
    data_array[0] = temp;
  }
  if (typeof to_operation == 'number') {
    const temp = to_operation;
    to_operation = [];
    to_operation[0] = temp;
  }
  let res: any[] = [];
  let safety = [];

  function get_toStore_object(data: NumTsMatrix[], replace: boolean) {
    let safety = [];
    if (replace == true) {
      safety = data;
    } else {
      safety = deepCopy(data);
    }
    return safety;
  }

  if (NumTS.is_first_greater(data_array, to_operation)) {
    safety = get_toStore_object(data_array, replace);
    res = NumTS.innerExecute(safety, to_operation, task_to_perform(false), res);
  } else {
    safety = get_toStore_object(to_operation, replace);
    res = NumTS.innerExecute(safety, data_array, task_to_perform(true), res);
  }
  return res;
};

NumTS.subtract = (a: NumTsMatrix[] | NumTsMatrix, b: NumTsMatrix[] | NumTsMatrix, replace = false) => {
  const gen_inner_subtract = (shuffle: boolean) => (a: number, b: number) => {
    /**
     * inner_subtract.shuffle is set by the execute function of operation
     */
    if (shuffle) {
      return b - a;
    } else {
      return a - b;
    }
  };
  return NumTS.mainExecute(a as NumTsMatrix[], b as NumTsMatrix[], replace as boolean, gen_inner_subtract);
};

NumTS.multiply = (a: NumTsMatrix[] | NumTsMatrix, b: NumTsMatrix[] | NumTsMatrix, replace = false) => {
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const gen_inner_multiply = (_: boolean) => (a: number, b: number) => {
    return b * a;
  };
  return NumTS.mainExecute(a as NumTsMatrix[], b as NumTsMatrix[], replace as boolean, gen_inner_multiply);
};

NumTS.power = (a: NumTsMatrix[] | NumTsMatrix, b: NumTsMatrix[] | NumTsMatrix, replace = false) => {
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const gen_inner_power = (shuffle: boolean) => (a: number, b: number) => {
    /**
     * inner_power.shuffle is set by the execute function of operation
     */

    if (shuffle) {
      return Math.pow(b, a);
    }
    return Math.pow(a, b);
  };
  return NumTS.mainExecute(a as NumTsMatrix[], b as NumTsMatrix[], replace as boolean, gen_inner_power);
};

NumTS.squareRoot = (data: NumTsMatrix[]) => NumTS.power(data, 1 / 2);

NumTS.cubeRoot = (data: NumTsMatrix[]) => NumTS.power(data, 1 / 3);

NumTS.nThRoot = (data: NumTsMatrix[], raiseTo: NumTsMatrix) => {
  function innerNThRoot(raiseTo: NumTsMatrix) {
    if (typeof raiseTo == 'number') {
      return 1 / raiseTo;
    }

    raiseTo.forEach((_, j) => {
      raiseTo[j] = innerNThRoot(raiseTo[j]);
    });
    return raiseTo as unknown as number;
  }

  const d = innerNThRoot(raiseTo);

  return NumTS.power(data, d);
};

NumTS.exp = (a: NumTsMatrix) => NumTS.power(2.71828, a);

function getBaseLog(x: number, y: number) {
  /**
   * x is base
   * y is the value of which log is required
   */
  return Math.log(y) / Math.log(x);
}

NumTS.log10 = (a: NumTsMatrix[] | NumTsMatrix, replace = false) => {
  const inner_log = (_: boolean) => (a: number, b: number) => {
    return Number(getBaseLog(b, a));
  };
  return NumTS.mainExecute(a as NumTsMatrix[], 10 as unknown as NumTsMatrix[], replace, inner_log);
};

NumTS.logE = (a: NumTsMatrix[] | NumTsMatrix, replace = false) => {
  const inner_log = (_: boolean) => (a: number, b: number) => {
    return Number(getBaseLog(b, a));
  };
  return NumTS.mainExecute(a as NumTsMatrix[], 2.718281828459045 as unknown as NumTsMatrix[], replace, inner_log);
};

NumTS.log = (a: NumTsMatrix[] | NumTsMatrix, b: NumTsMatrix[] | NumTsMatrix, replace = false) => {
  const inner_log = (shuffle: boolean) => (a: number, b: number) => {
    if (shuffle) {
      return Number(getBaseLog(a, b));
    } else {
      return Number(getBaseLog(b, a));
    }
  };
  return NumTS.mainExecute(a as NumTsMatrix[], b as NumTsMatrix[], replace, inner_log);
};

NumTS.add = (a: NumTsMatrix[] | NumTsMatrix, b: NumTsMatrix[] | NumTsMatrix, replace = false) => {
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const gen_inner_add = (_: boolean) => (a: number, b: number) => {
    return b + a;
  };
  return NumTS.mainExecute(a as NumTsMatrix[], b as NumTsMatrix[], replace as boolean, gen_inner_add);
};

NumTS.divide = (a: NumTsMatrix[] | NumTsMatrix, b: NumTsMatrix[] | NumTsMatrix, replace = false) => {
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const gen_inner_divide = (shuffle: boolean) => (a: number, b: number) => {
    if (shuffle) {
      return b / a;
    } else {
      return a / b;
    }
  };
  return NumTS.mainExecute(a as NumTsMatrix[], b as NumTsMatrix[], replace as boolean, gen_inner_divide);
};

NumTS.findPopulationStandardDeviation = (data: NumTsMatrix[]) => {
  const mean: number = NumTS.findMean(data) as number;
  const temp = NumTS.subtract(data, mean);
  return Math.sqrt(NumTS.findMean(NumTS.findSquare(temp) as NumTsMatrix[]));
};

NumTS.findPopulationVariance = (data: NumTsMatrix[]) => {
  const temp = NumTS.findPopulationStandardDeviation(data);
  return temp * temp;
};

NumTS.findSampleStandardDeviation = (data: NumTsMatrix[]) => {
  const mean = NumTS.findMean(data);

  const temp = NumTS.subtract(data, mean);
  const total = NumTS.findTotalElements(temp);
  return Math.sqrt((NumTS.findMean(temp) * total) / (total - 1));
};

NumTS.findSampleVariance = (data: NumTsMatrix[]) => {
  const temp = NumTS.findSampleStandardDeviation(data);
  return temp * temp;
};

NumTS.findAllFrequency = (data: NumTsMatrix[]) => {
  function inner_findAllFrequency(wholeData: NumTsMatrix[], data: NumTsMatrix[], ans = {}) {
    // var ans = {};
    data.forEach((_, i) => {
      if (typeof data[i] == 'number') {
        // eslint-disable-next-line @typescript-eslint/ban-ts-comment
        // @ts-ignore
        ans[data[i] as keyof typeof ans] = NumTS.findFrequency(wholeData, data[i] as number);
      } else {
        inner_findAllFrequency(wholeData, data[i] as NumTsMatrix[], ans);
      }
    });
  }
  const ans = {};
  inner_findAllFrequency(data, data, ans);
  return ans;
};

NumTS.findMode = (data: NumTsMatrix[]) => {
  function check_if_all_same_frequency(data: NumTsMatrix[]) {
    const check = data[Object.keys(data)[0] as keyof typeof data];
    for (const v of Object.keys(data)) {
      if (data[v as keyof typeof data] != check) {
        return false;
      }
    }
    return true;
  }

  const res = NumTS.findAllFrequency(data);
  if (check_if_all_same_frequency(res as NumTsMatrix[])) {
    return undefined;
  } else {
    let max = res[Object.keys(res)[0] as keyof typeof res] as number;
    for (const v of Object.keys(res)) {
      if (res[v as keyof typeof res] > max) {
        max = res[v as keyof typeof res] as number;
      }
    }
    const ans = [];
    let counter = -1;
    for (const v of Object.keys(res)) {
      if (res[v as keyof typeof res] == max) {
        ans[++counter] = Number(v);
      }
    }
    return ans;
  }
};
