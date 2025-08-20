#!/bin/bash
git push --delete origin v1.0.0
git tag -d v1.0.0
git tag v1.0.0
git push origin v1.0.0
