# CMake generated Testfile for 
# Source directory: /student/wuhungma/CSC358/pa1_starter_code
# Build directory: /student/wuhungma/CSC358/pa1_starter_code/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[compile with bug-checkers]=] "/usr/local/lib/python3.10/dist-packages/cmake/data/bin/cmake" "--build" "/student/wuhungma/CSC358/pa1_starter_code/build" "-t" "functionality_testing" "webget")
set_tests_properties([=[compile with bug-checkers]=] PROPERTIES  FIXTURES_SETUP "compile" TIMEOUT "-1" _BACKTRACE_TRIPLES "/student/wuhungma/CSC358/pa1_starter_code/etc/tests.cmake;6;add_test;/student/wuhungma/CSC358/pa1_starter_code/etc/tests.cmake;0;;/student/wuhungma/CSC358/pa1_starter_code/CMakeLists.txt;16;include;/student/wuhungma/CSC358/pa1_starter_code/CMakeLists.txt;0;")
add_test([=[t_webget]=] "/student/wuhungma/CSC358/pa1_starter_code/tests/webget_t.sh" "/student/wuhungma/CSC358/pa1_starter_code/build")
set_tests_properties([=[t_webget]=] PROPERTIES  FIXTURES_REQUIRED "compile" _BACKTRACE_TRIPLES "/student/wuhungma/CSC358/pa1_starter_code/etc/tests.cmake;17;add_test;/student/wuhungma/CSC358/pa1_starter_code/etc/tests.cmake;0;;/student/wuhungma/CSC358/pa1_starter_code/CMakeLists.txt;16;include;/student/wuhungma/CSC358/pa1_starter_code/CMakeLists.txt;0;")
add_test([=[net_interface_test_typical]=] "/student/wuhungma/CSC358/pa1_starter_code/build/tests/net_interface_test_typical_sanitized")
set_tests_properties([=[net_interface_test_typical]=] PROPERTIES  FIXTURES_REQUIRED "compile" _BACKTRACE_TRIPLES "/student/wuhungma/CSC358/pa1_starter_code/etc/tests.cmake;10;add_test;/student/wuhungma/CSC358/pa1_starter_code/etc/tests.cmake;22;ttest;/student/wuhungma/CSC358/pa1_starter_code/etc/tests.cmake;0;;/student/wuhungma/CSC358/pa1_starter_code/CMakeLists.txt;16;include;/student/wuhungma/CSC358/pa1_starter_code/CMakeLists.txt;0;")
add_test([=[net_interface_test_reply]=] "/student/wuhungma/CSC358/pa1_starter_code/build/tests/net_interface_test_reply_sanitized")
set_tests_properties([=[net_interface_test_reply]=] PROPERTIES  FIXTURES_REQUIRED "compile" _BACKTRACE_TRIPLES "/student/wuhungma/CSC358/pa1_starter_code/etc/tests.cmake;10;add_test;/student/wuhungma/CSC358/pa1_starter_code/etc/tests.cmake;23;ttest;/student/wuhungma/CSC358/pa1_starter_code/etc/tests.cmake;0;;/student/wuhungma/CSC358/pa1_starter_code/CMakeLists.txt;16;include;/student/wuhungma/CSC358/pa1_starter_code/CMakeLists.txt;0;")
add_test([=[net_interface_test_learn]=] "/student/wuhungma/CSC358/pa1_starter_code/build/tests/net_interface_test_learn_sanitized")
set_tests_properties([=[net_interface_test_learn]=] PROPERTIES  FIXTURES_REQUIRED "compile" _BACKTRACE_TRIPLES "/student/wuhungma/CSC358/pa1_starter_code/etc/tests.cmake;10;add_test;/student/wuhungma/CSC358/pa1_starter_code/etc/tests.cmake;24;ttest;/student/wuhungma/CSC358/pa1_starter_code/etc/tests.cmake;0;;/student/wuhungma/CSC358/pa1_starter_code/CMakeLists.txt;16;include;/student/wuhungma/CSC358/pa1_starter_code/CMakeLists.txt;0;")
add_test([=[net_interface_test_pending]=] "/student/wuhungma/CSC358/pa1_starter_code/build/tests/net_interface_test_pending_sanitized")
set_tests_properties([=[net_interface_test_pending]=] PROPERTIES  FIXTURES_REQUIRED "compile" _BACKTRACE_TRIPLES "/student/wuhungma/CSC358/pa1_starter_code/etc/tests.cmake;10;add_test;/student/wuhungma/CSC358/pa1_starter_code/etc/tests.cmake;25;ttest;/student/wuhungma/CSC358/pa1_starter_code/etc/tests.cmake;0;;/student/wuhungma/CSC358/pa1_starter_code/CMakeLists.txt;16;include;/student/wuhungma/CSC358/pa1_starter_code/CMakeLists.txt;0;")
add_test([=[net_interface_test_expiry]=] "/student/wuhungma/CSC358/pa1_starter_code/build/tests/net_interface_test_expiry_sanitized")
set_tests_properties([=[net_interface_test_expiry]=] PROPERTIES  FIXTURES_REQUIRED "compile" _BACKTRACE_TRIPLES "/student/wuhungma/CSC358/pa1_starter_code/etc/tests.cmake;10;add_test;/student/wuhungma/CSC358/pa1_starter_code/etc/tests.cmake;26;ttest;/student/wuhungma/CSC358/pa1_starter_code/etc/tests.cmake;0;;/student/wuhungma/CSC358/pa1_starter_code/CMakeLists.txt;16;include;/student/wuhungma/CSC358/pa1_starter_code/CMakeLists.txt;0;")
add_test([=[net_interface_test_independence]=] "/student/wuhungma/CSC358/pa1_starter_code/build/tests/net_interface_test_independence_sanitized")
set_tests_properties([=[net_interface_test_independence]=] PROPERTIES  FIXTURES_REQUIRED "compile" _BACKTRACE_TRIPLES "/student/wuhungma/CSC358/pa1_starter_code/etc/tests.cmake;10;add_test;/student/wuhungma/CSC358/pa1_starter_code/etc/tests.cmake;27;ttest;/student/wuhungma/CSC358/pa1_starter_code/etc/tests.cmake;0;;/student/wuhungma/CSC358/pa1_starter_code/CMakeLists.txt;16;include;/student/wuhungma/CSC358/pa1_starter_code/CMakeLists.txt;0;")
subdirs("util")
subdirs("src")
subdirs("tests")
