/*

Copyright (c) 2009-2018, Thanos Theo. All rights reserved.
Released Under a Simplified BSD (FreeBSD) License
for academic, personal & non-commercial use.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those
of the author and should not be interpreted as representing official policies,
either expressed or implied, of the FreeBSD Project.

A Commercial License is also available for commercial use with
special restrictions and obligations at a one-off fee. See links at:
1. http://www.dotredconsultancy.com/openglrenderingenginetoolrelease.php
2. http://www.dotredconsultancy.com/openglrenderingenginetoolsourcecodelicence.php
Please contact Thanos Theo (thanos.theo@dotredconsultancy.com) for more information.

*/

#pragma once

#ifndef __ConcurrentBlockingQueue_h
#define __ConcurrentBlockingQueue_h

#include <mutex>
#include <memory>
#include <queue>
#include <condition_variable>

/** @brief Namespace Utils contains utility classes with mainly static CPU related methods.
* @author Thanos Theo, 2009-2018
* @version 14.0.0.0
*/
namespace Utils
{
  /** @brief Namespace CPUParallelism encapsulates usage of the N-CP parallelism idea.
  *
  *  This namespace encapsulates usage of the N-CP parallelism idea.\n
  *  CPUParallelism libraries originally based on with further extensions: http://www.manning.com/williams/.\n
  *  The N-CP idea was based on: http://www.biolayout.org/wp-content/uploads/2013/01/Manuscript.pdf.\n
  *  Further inspiration was found here: http://jcip.net.s3-website-us-east-1.amazonaws.com/.\n
  *
  * @author Thanos Theo, 2009-2018
  * @version 14.0.0.0
  */
  namespace CPUParallelism
  {
    /** @brief This class encapsulates usage of a concurrent blocking queue.
    *
    *  ConcurrentBlockingQueue.h:
    *  =========================
    *  This class encapsulates usage of a concurrent blocking queue.\n
    *  CPUParallelism libraries originally based on with further extensions: http://www.manning.com/williams/.\n
    *  The N-CP idea was based on: http://www.biolayout.org/wp-content/uploads/2013/01/Manuscript.pdf.\n
    *  Further inspiration was found here: http://jcip.net.s3-website-us-east-1.amazonaws.com/.\n
    *
    * @author Thanos Theo, 2009-2018
    * @version 14.0.0.0
    */
    template<typename T> class ConcurrentBlockingQueue final
    {
    public:

      ConcurrentBlockingQueue()  = default;
      ~ConcurrentBlockingQueue() = default; // no virtual destructor for data-oriented design (no up-casting should ever be used)
      ConcurrentBlockingQueue(const ConcurrentBlockingQueue& other) = delete; // copy-constructor deleted
      ConcurrentBlockingQueue(ConcurrentBlockingQueue&& other)      = delete; // move-constructor deleted
      ConcurrentBlockingQueue& operator=(const ConcurrentBlockingQueue& other) = delete; //      assignment operator deleted
      ConcurrentBlockingQueue& operator=(ConcurrentBlockingQueue&& other)      = delete; // move-assignment operator deleted

      void waitAndPop(T& value)
      {
        std::unique_lock<std::mutex> lockDataMutex(dataMutex_);
        dataCondition_.wait(lockDataMutex, [this] { return !dataQueue_.empty(); });
        value = std::move(*dataQueue_.front());
        dataQueue_.pop();
      }

      bool tryPop(T& value)
      {
        std::lock_guard<std::mutex> lockDataMutex(dataMutex_);
        if (dataQueue_.empty())
        {
          return false;
        }
        else
        {
          value = std::move(*dataQueue_.front());
          dataQueue_.pop();

          return true;
        }
      }

      std::shared_ptr<T> waitAndPop()
      {
        std::unique_lock<std::mutex> lockDataMutex(dataMutex_);
        dataCondition_.wait(lockDataMutex, [this] { return !dataQueue_.empty(); });
        std::shared_ptr<T> result = dataQueue_.front();
        dataQueue_.pop();

        return result;
      }

      std::shared_ptr<T> tryPop()
      {
        std::lock_guard<std::mutex> lockDataMutex(dataMutex_);
        if (dataQueue_.empty())
        {
          return std::shared_ptr<T>();
        }

        std::shared_ptr<T> result = dataQueue_.front();
        dataQueue_.pop();

        return result;
      }

      void emplace(T newValue)
      {
        std::shared_ptr<T> data(std::make_shared<T>(std::move(newValue)));
        std::lock_guard<std::mutex> lockDataMutex(dataMutex_);
        dataQueue_.emplace(std::move(data));
        dataCondition_.notify_one();
      }

      bool empty() const
      {
        std::lock_guard<std::mutex> lockDataMutex(dataMutex_);

        return dataQueue_.empty();
      }

    private:

      mutable std::mutex             dataMutex_;
      std::queue<std::shared_ptr<T>> dataQueue_;
      std::condition_variable        dataCondition_;
    };
  }
}

#endif // __ConcurrentBlockingQueue_h