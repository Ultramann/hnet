module Main where

import Neural.Matrix
import Numeric.LinearAlgebra
import Test.Hspec

{- Simple Net
   1
      [3, 2, -2] -> 1  |-1.5| = -0.5 -> 0                 bias
   2                    bias              [-2, 0.5] -> 7 | -3 | -> 4
      [-1, 2, 3] -> 12 | +2 | =  14 -> 14
   3
-}

zeroVec = vector [0.0,0.0]
zeroNet = [(relu, zeroVec, matrix 2 [0,0,0,0]), (relu, vector [0], matrix 1 [0,0])]
vTest = vector [1, 2, 3]
mTest = matrix 2 [3.0, -1.0, 2.0, 2.0, -2.0, 3.0]
simpleNet = [(relu, vector [-1.5, 2.0], mTest), (relu, vector [-3.0], matrix 1 [-2.0, 0.5])]

main :: IO ()
main = hspec $ do
  describe "Check feed layer" $ do
    it "passes trivial case" $ do
      (feedLayer (zeroVec, zeroVec) (head zeroNet)) `shouldBe` (zeroVec, zeroVec)
    it "passes simple case" $ do
      (feedLayer (vTest, vTest) (head simpleNet)) `shouldBe` (vector [-0.5,14.0], vector [0.0,14.0])

  describe "Check forward pass" $ do
    it "passes trivial case" $ do
      let zp = [(zeroVec, zeroVec), (zeroVec, zeroVec),
                (vector [0.0], vector [0.0])]
      (forwardPass zeroVec zeroNet) `shouldBe` zp
    it "passes simple case" $ do
      let fp = [(vector [1.0,2.0,3.0], vector [1.0,2.0,3.0]),
                (vector [-0.5,14.0], vector [0.0,14.0]),
                (vector [4.0], vector [4.0])]
      (forwardPass vTest simpleNet) `shouldBe` fp

  describe "Check chain" $ do
    it "passes trivial case" $ do
      let (_, _, zm) = last zeroNet
      let ze = vector [0.0]
      (chain (zeroVec, zm) ze) `shouldBe` zeroVec
    it "passes simple case" $ do
      let tl = vector [-0.5,14]
      let (_, _, lw) = last simpleNet
      let e = vector [0.5]
      let chained = vector [0,0.25]
      (chain (tl, lw) e) `shouldBe` chained

  describe "Check back prop" $ do
    it "passes simple case" $ do
      let tls = [zeroVec, vector [0]]
      let lws = map (\(_, _, x) -> x) zeroNet
      (backProp (vector [0]) tls lws) `shouldBe` tls
