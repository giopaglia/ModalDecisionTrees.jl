# using Logging
# @error "aoe"
# throw_n_log("aoe")
# throw_n_log("aoe")
using Logging, LoggingExtras
using Telegram, Telegram.API
using ConfigEnv

dotenv()

# Log to Telegram Bot!!!
tg = TelegramClient()
tg_logger = TelegramLogger(tg; async = false)
demux_logger = TeeLogger(
	MinLevelLogger(tg_logger, Logging.Error),
	ConsoleLogger()
)
global_logger(demux_logger)

run(`julia -i -t16 $(ARGS[1])`)

# # Echo bot
# run_bot() do msg
# 	println(msg.message.text)
# 	sendMessage(text = msg.message.text, chat_id = msg.message.chat.id)
# end

