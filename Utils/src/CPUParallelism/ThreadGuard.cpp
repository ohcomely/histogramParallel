#include "CPUParallelism/ThreadGuard.h"
#include "UtilityFunctions.h"

using namespace std;
using namespace Utils::CPUParallelism;

// member initialization list has the thread object being last as it may start running a function immediately
ThreadGuard::ThreadGuard(thread&& thread, DestructorAction action) noexcept
  : action_{action}
  , thread_{move(thread)}
{
  if (action_ == DestructorAction::JOIN && !thread_.joinable())
  {
    DebugConsole_consoleOutLine("ThreadGuard constructor: Not a joinable() thread.");
    action_ = DestructorAction::DETACH;
  }
}

ThreadGuard::~ThreadGuard() noexcept
{
  if (thread_.joinable())
  {
    if (action_ == DestructorAction::JOIN)
    {
      thread_.join();
    }
    else // if (action_ == DestructorAction::DETACH)
    {
      thread_.detach();
    }
  }
}