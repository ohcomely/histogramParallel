#include "Randomizers.h"
#include "AccurateTimers.h"

using namespace std;
using namespace Utils;
using AccurateTimer = AccurateTimers::AccurateCPUTimer;

Randomizers::UniformRandom::UniformRandom() noexcept
{
  rng_.seed(AccurateTimer::getNanosecondsTimeSinceEpoch());
}

Randomizers::RandomRNGWELL512::RandomRNGWELL512() noexcept
{
  UniformRandom random;
  for (size_t i = 0; i < STATE_SIZE; ++i)
  {
    state_[i] = random.getUniformInteger(); // initialize state to random bits
  }
}

uint64_t Randomizers::RandomRNGWELL512::getRandomInteger()
{
  uint64_t a, b, c, d;
  a  = state_[index_];
  c  = state_[(index_ + 13) & 15];
  b  = a ^ c ^ (a << 16) ^ (c << 15);
  c  = state_[(index_ + 9) & 15];
  c ^= (c >> 11);
  a  = state_[index_] = b ^ c;
  d  = a ^ ((a << 5) & 0xDA442D20UL);
  index_ = (index_ + 15) & 15;
  a  = state_[index_];
  state_[index_] = a ^ b ^ d ^ (a << 2) ^ (b << 18) ^ (c << 28);

  return state_[index_];
}