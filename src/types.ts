export type NumTsMatrix = number | number[] | NumTsMatrix[];

export type Zeros = (shape: number | number[]) => NumTsMatrix;

export type FillFunction = (this: any, indexes: number[]) => number;

export type EyePros = { column?: number; diagonal?: number };

export type ProdProps = {
  axis?: number;
  initial?: number;
  where?: boolean[];
  flat?: boolean;
};
