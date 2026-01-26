#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <nshogi/core/initializer.h>
#include <nshogi/io/sfen.h>

#include "types.h"
#include "mcts.h"
#include "nn.h"
#include "encoder.h"

namespace py = pybind11;

PYBIND11_MODULE(mcts_cpp, m) {
        nshogi::core::initializer::initializeAll();

        m.def("encode_state_mirror", &encode_state_mirror, "Get both normal and mirrored state tensors");
        m.def("get_mirrored_move_index", &get_mirrored_move_index, "convert move to mirrored index");

        
        py::class_<MCTS<true>::SearchResult>(m, "SearchResult")
                .def_readonly("best_move", &MCTS<true>::SearchResult::best_move)
                .def_readonly("visit_counts", &MCTS<true>::SearchResult::visit_counts)
                .def_readonly("value", &MCTS<true>::SearchResult::root_value);
        
        py::class_<nshogi::core::Move32>(m, "Move32")
                .def(py::init<>())
                .def("__str__", [](const nshogi::core::Move32& m) {
                        std::stringstream ss;
                        ss << nshogi::io::sfen::move32ToSfen(m); 
                        return ss.str();
                });
        
        py::class_<GameState>(m, "GameState")
        .def("__str__", [](const GameState& state) {
                std::stringstream ss;
                ss << nshogi::io::sfen::stateToSfen(*(state.state)); 
                return ss.str();
        })
        .def_static("initial", []() {
                return GameState(nshogi::io::sfen::StateBuilder::getInitialState());
        })
        .def("is_terminal", &GameState::is_terminal)
        .def("result", &GameState::result)
        .def("legal_moves", &GameState::legal_moves)
        .def("do_move", &GameState::do_move);
        
        py::class_<NeuralNetwork, std::shared_ptr<NeuralNetwork>>(m, "NeuralNetwork")
        .def(py::init<const std::string&,const std::string&, int>(), py::arg("path"), py::arg("device"), py::arg("queue_size") = 16)
        .def("evaluate", &NeuralNetwork::evaluate_async)
        .def("get_encoded_state", [](NeuralNetwork &self, const GameState &state) {
                        return encode_state(state);
        }, py::arg("state"));
        py::class_<NodePool<true>>(m, "NodePool")
                .def(py::init<>())
                .def("reset", &NodePool<true>::reset);
                
        py::class_<MCTS<true>>(m, "JAMES_trainer")
                .def(py::init<NodePool<true>&, int>(), py::arg("pool"), py::arg("num_threads"))
                .def("start_new_game", &MCTS<true>::start_new_game)
                .def("update_root", &MCTS<true>::update_root)
                .def("search", &MCTS<true>::search, py::arg("state"), py::arg("nn"), py::arg("iterations"),
                py::return_value_policy::reference, 
                py::call_guard<py::gil_scoped_release>(),
                "Search the game tree");

        py::class_<MCTS<false>>(m, "JAMES")
                .def(py::init<NodePool<false>&, int>(), py::arg("pool"), py::arg("num_threads"))
                .def("start_new_game", &MCTS<false>::start_new_game)
                .def("update_root", &MCTS<false>::update_root)
                .def("search", &MCTS<false>::search, py::arg("state"), py::arg("nn"), py::arg("iterations"),
                py::return_value_policy::reference, 
                py::call_guard<py::gil_scoped_release>(),
                "Search the game tree");
}
